# 表面贴合会话:把设置解析成对应关系与影响权重,驱动全部数据类型的传输。
# 架构:一次会话只构建一次顶点域/角点域对应关系,全部数据类型复用同一内核采样。
# 映射方式与 Blender DataTransfer 修改器逐项对齐(标识符/命名/语义一致):
#   顶点域: TOPOLOGY / NEAREST / EDGE_NEAREST / EDGEINTERP_NEAREST /
#           POLY_NEAREST / POLYINTERP_NEAREST / POLYINTERP_VNORPROJ (+ 本插件独有 UV)
#   角点域: TOPOLOGY / NEAREST_NORMAL / NEAREST_POLYNOR / NEAREST_POLY /
#           POLYINTERP_NEAREST / POLYINTERP_LNORPROJ (+ 本插件独有 UV)
# 影响权重统一管线:mix × 顶点组遮罩 × 选择遮罩 × 距离衰减 × 命中有效性,
# 所有数据按 result = existing + (sampled - existing) × influence 混合落地。

import numpy as np

from .channels import (
    ChannelError,
    DIRECTIONAL_CHANNELS,
    NAMED_CHANNELS,
    POSITION,
    POSITIONAL_CHANNELS,
    UV,
    TargetChannel,
    adapt_components,
    channel_label,
    channel_names,
    default_target_name,
    lift_to_match_space,
    read_source_channel,
    remap_values,
    resolve_match_name,
    resolve_source_name,
)
from .correspondence import (
    CORNER,
    POINT,
    SurfaceCorrespondence,
    EdgeNearestQuery,
    GatherCorrespondence,
    DirectVertexCorrespondence,
    CombinedVertexCorrespondence,
    SourceDomainBridge,
    TopologyVertexCorrespondence,
    DirectCornerCorrespondence,
    TopologyCornerCorrespondence,
    deduplicate_queries,
    exact_value_groups,
    match_corners_by_face_values,
    snap_positions_to_nearest,
    build_kd_tree,
    query_kd_nearest,
    ragged_arange,
    segment_best_rows,
)
from .mesh_buffers import (
    MeshBufferSnapshot,
    add_numbered_shape_key,
    add_numbered_uv_layer,
    apply_vertex_positions,
    read_shape_key_mix_positions,
    matrix_to_numpy,
    transform_points,
    transform_directions,
    normalized_rows,
    read_uv_layer,
    write_uv_layer,
    ensure_uv_layer,
    read_color_attribute,
    ensure_color_attribute,
    write_color_attribute,
    ensure_shape_key,
    read_shape_key_positions,
    write_shape_key_positions,
    write_corner_normals,
    write_vertex_group_weights,
)
from .properties import resolved_corner_mapping, resolved_vertex_mapping
from .shape_key_drivers import transfer_shape_key_drivers

# 与目标几何无关的映射(可在 Shape 写回后惰性构建)。
_POSITION_INDEPENDENT_MAPPINGS = {'TOPOLOGY', 'UV'}


class ConformError(RuntimeError):
    """配置或数据不满足传输前提时抛出,由算子层转成 report。"""


class ConformSession:
    def __init__(self, context, settings, source_object, target_object):
        self._context = context
        self.settings = settings
        self.source_object = source_object
        self.target_object = target_object

        if source_object is None or source_object.type != 'MESH':
            raise ConformError("Source must be a mesh object")
        if target_object is None or target_object.type != 'MESH':
            raise ConformError("Target must be a mesh object")
        # 同一物体 = 就地转换(把自己的某个通道变成另一个通道),按序号一一对应,
        # 精确且不需要任何空间查询,所以强制走拓扑映射。
        self.same_object = source_object == target_object

        # 目标一律按"当前可见形状"(形态键混合后)参与匹配与混合,这样每次应用都
        # 接着上一次的结果继续推进,而不是每次都从静止态重来。
        # 它会短暂增删一个形态键,必须赶在源求值网格建立之前做完。
        target_mix = read_shape_key_mix_positions(target_object)
        depsgraph = context.evaluated_depsgraph_get() if settings.use_evaluated_source else None
        self.source_snapshot = MeshBufferSnapshot(
            source_object, settings.use_evaluated_source, depsgraph)
        self.target_snapshot = MeshBufferSnapshot(target_object)
        if target_mix is not None:
            self.target_snapshot.seed_vertex_positions(target_mix)

        if self.target_snapshot.vertex_count == 0:
            raise ConformError("Target mesh has no vertices")

        self._source_matrix = matrix_to_numpy(source_object.matrix_world)
        self._target_matrix = matrix_to_numpy(target_object.matrix_world)
        identity = np.identity(4, dtype=np.float64)
        if settings.transform_space == 'WORLD':
            try:
                target_inverse = np.linalg.inv(self._target_matrix)
            except np.linalg.LinAlgError:
                raise ConformError("Target matrix is not invertible (zero scale?)")
            # 位置数据映射:源局部 → 世界 → 目标局部。
            self._position_matrix = target_inverse @ self._source_matrix
            # 匹配空间(世界)→ 目标局部,供通道转换里"任意数据当坐标写回"用。
            self._match_to_target_matrix = target_inverse
        else:
            self._position_matrix = identity
            self._match_to_target_matrix = identity

        self._source_bridge = None
        self._match_cache = {}
        self._vertex_correspondence = None
        self._corner_correspondence = None
        self._vertex_influence_base = None
        self._influence_cache = {}
        self._exact_match_counts = {}
        self.warnings = []
        self.summaries = []

    def free(self):
        self.source_snapshot.free()
        self.target_snapshot.free()

    # ==================== 匹配空间几何 ====================

    @property
    def source_bridge(self):
        """源域桥:让任意一种对应关系都能采样任意一个域的源数据(懒建一次)。"""
        if self._source_bridge is None:
            self._source_bridge = SourceDomainBridge(
                self.source_snapshot.loop_vertex_indices,
                self.source_snapshot.vertex_count)
        return self._source_bridge

    def vertex_mapping(self):
        """预设解析后的顶点域映射;就地转换恒为拓扑(逐序号,精确)。"""
        if self.same_object:
            return 'TOPOLOGY'
        return resolved_vertex_mapping(self.settings)

    def corner_mapping(self):
        if self.same_object:
            return 'TOPOLOGY'
        return resolved_corner_mapping(self.settings)

    def _is_world_space(self):
        return self.settings.transform_space == 'WORLD'

    def _search_max_distance(self):
        return self.settings.max_distance if self.settings.use_max_distance else None

    def _cached_match(self, key, builder):
        value = self._match_cache.get(key)
        if value is None:
            value = builder()
            self._match_cache[key] = value
        return value

    def _get_source_match_positions(self):
        def build():
            positions = self.source_snapshot.vertex_positions
            if self._is_world_space():
                positions = transform_points(positions, self._source_matrix)
            return positions
        return self._cached_match("source_positions", build)

    def _get_target_match_positions(self):
        def build():
            positions = self.target_snapshot.vertex_positions
            if self._is_world_space():
                positions = transform_points(positions, self._target_matrix)
            return positions
        return self._cached_match("target_positions", build)

    def _match_space_source_normals(self, normals):
        if not self._is_world_space():
            return normals
        inverse_transpose = np.linalg.inv(self._source_matrix[:3, :3]).T
        return normalized_rows(transform_directions(normals, inverse_transpose))

    def _match_space_target_normals(self, normals):
        if not self._is_world_space():
            return normals
        inverse_transpose = np.linalg.inv(self._target_matrix[:3, :3]).T
        return normalized_rows(transform_directions(normals, inverse_transpose))

    def _get_surface_3d(self):
        """源表面 3D BVH。WORLD 模式在世界空间建树,距离阈值即世界单位。"""
        def build():
            source = self.source_snapshot
            if len(source.mesh.polygons) == 0:
                raise ConformError("Source mesh has no faces for face mapping")
            return SurfaceCorrespondence(
                self._get_source_match_positions(),
                source.triangle_vertex_indices,
                source.triangle_vertex_indices,
                source.triangle_loop_indices)
        return self._cached_match("surface_3d", build)

    def _get_edge_query(self):
        def build():
            edges = self.source_snapshot.edge_vertex_indices
            if edges.shape[0] == 0:
                raise ConformError("Source mesh has no edges for edge mapping")
            return EdgeNearestQuery(self._get_source_match_positions(), edges)
        return self._cached_match("edge_query", build)

    def _get_source_vertex_kd(self):
        def build():
            if self.source_snapshot.vertex_count == 0:
                raise ConformError("Source mesh has no vertices")
            return build_kd_tree(self._get_source_match_positions())
        return self._cached_match("source_kd", build)

    def _read_basis_values(self, snapshot, kind, name, is_target):
        """把基准通道读成匹配空间里的 (N, 3) 点。返回 (值, 域, 层名, 分量数)。

        位置/方向类通道按各自物体的矩阵进匹配空间,其余通道原样使用 ——
        源与目标读的是同一个通道,量纲天然对齐。
        """
        try:
            channel_data = read_source_channel(
                snapshot, kind, name, side="Target" if is_target else "Source")
        except ChannelError as error:
            raise ConformError(f"Match basis — {error}")
        values = channel_data.values
        if channel_data.positional:
            matrix = self._target_matrix if is_target else self._source_matrix
            if self._is_world_space():
                values = transform_points(values, matrix)
        elif channel_data.directional:
            values = (self._match_space_target_normals(values) if is_target
                      else self._match_space_source_normals(values))
        return (lift_to_match_space(values), channel_data.domain,
                channel_data.name, channel_data.components)

    @staticmethod
    def _value_extent(values):
        """一组匹配空间点的包围盒对角线长度 = 这个基准"有多大"。"""
        if values.shape[0] == 0:
            return 0.0
        return float(np.linalg.norm(values.max(axis=0) - values.min(axis=0)))

    def _require_basis_extent(self, extent, kind, name, components, side):
        """基准整体塌成一个点时必须当场报错 —— 否则所有查询都命中同一处,
        表现就是目标网格被吸成一个点(旧版对 UV 有这道闸,泛化后必须保留)。"""
        if components < 2 or extent > 1e-12:
            return
        raise ConformError(
            f"{side} {channel_label(kind).lower()} '{name}' has no extent — every "
            f"element would match the same spot. Pick the layer you actually "
            f"unwrapped / painted")

    def _get_source_basis_values(self, kind):
        """源侧基准值(匹配空间,已升到 3 分量),供几何建树与精确值匹配共用。"""
        def build():
            return self._read_basis_values(
                self.source_snapshot, kind, self.settings.match_basis_name_source,
                is_target=False)
        return self._cached_match(("source_basis", kind), build)

    def _source_element_positions(self, domain):
        """源元素(角点域=loop,顶点域=vertex)各自的匹配空间位置,用于歧义决胜。"""
        positions = self._get_source_match_positions()
        if domain == CORNER:
            return positions[self.source_snapshot.loop_vertex_indices]
        return positions

    def _resolve_value_ambiguity(self, chosen, ambiguous_rows, member_offsets, members,
                                 target_group, element_positions, domain):
        """同一个基准值落在多个源元素上时,按目标自身的面上下文决胜。

        同面的其他角点已经唯一确定了落点,取离那个锚点最近的候选 —— 两个 UV 岛
        撞在同一坐标时,只有这样才选得回本来那一侧;逐元素单看是无解的。
        """
        if domain != CORNER:
            for row in ambiguous_rows.tolist():
                group = target_group[row]
                chosen[row] = members[member_offsets[group]]
            return chosen
        target = self.target_snapshot
        face_of_loop = target.loop_polygon_indices
        face_count = len(target.mesh.polygons)
        settled = chosen >= 0
        anchor_counts = np.bincount(
            face_of_loop[settled], minlength=face_count).astype(np.float64)
        settled_positions = element_positions[chosen[settled]]
        anchor_sums = np.zeros((face_count, settled_positions.shape[1]))
        for channel in range(settled_positions.shape[1]):
            anchor_sums[:, channel] = np.bincount(
                face_of_loop[settled], weights=settled_positions[:, channel],
                minlength=face_count)
        anchors = anchor_sums / np.maximum(anchor_counts, 1.0)[:, None]
        for row in ambiguous_rows.tolist():
            group = target_group[row]
            candidates = members[member_offsets[group]:member_offsets[group + 1]]
            face = face_of_loop[row]
            if anchor_counts[face] > 0.0:
                offsets = element_positions[candidates] - anchors[face]
                chosen[row] = candidates[
                    np.argmin(np.einsum('ij,ij->i', offsets, offsets))]
            else:
                chosen[row] = candidates[0]
        return chosen

    def _apply_exact_basis_matches(self, rows, surface, kind, domain, target_values):
        """基准值与源逐位相同的目标元素,直接对到那个源元素,不走几何查询。

        源 UV 有重叠时同一坐标被多个源三角形覆盖、命中距离全为 0,几何查询挑中哪个
        纯看 BVH 遍历顺序,必然采错一部分;值相同就是同一处,先按值定死才是对的。
        一个值仍对应多个源元素时按面上下文决胜(见 _resolve_value_ambiguity)。
        """
        source_values, _domain, _name, _components = self._get_source_basis_values(kind)
        member_offsets, members, target_group = exact_value_groups(
            source_values, target_values)
        found = target_group >= 0
        if not np.any(found):
            return rows
        safe_group = np.where(found, target_group, 0)
        candidate_counts = np.where(
            found, member_offsets[safe_group + 1] - member_offsets[safe_group], 0)
        if domain == CORNER:
            # 先按整张面的值组合配对 —— 判据比单个值强得多,点级撞车在这里就解开了。
            chosen = match_corners_by_face_values(
                source_values, self.source_snapshot.polygon_loop_starts,
                self.source_snapshot.polygon_loop_totals,
                target_values, self.target_snapshot.polygon_loop_starts,
                self.target_snapshot.polygon_loop_totals,
                target_group.shape[0])
        else:
            chosen = np.full(target_group.shape[0], -1, dtype=np.int64)
        pending = chosen < 0
        unique_hit = pending & found & (candidate_counts == 1)
        chosen[unique_hit] = members[member_offsets[target_group[unique_hit]]]
        ambiguous_rows = np.nonzero(pending & found & (candidate_counts > 1))[0]
        if ambiguous_rows.shape[0]:
            chosen = self._resolve_value_ambiguity(
                chosen, ambiguous_rows, member_offsets, members, target_group,
                self._source_element_positions(domain), domain)

        slots = surface.element_slots(domain, source_values.shape[0])
        settled = chosen >= 0
        slot = np.full(chosen.shape[0], -1, dtype=np.int64)
        slot[settled] = slots[chosen[settled]]
        usable = slot >= 0
        if not np.any(usable):
            return rows
        self._exact_match_counts[kind] = (int(np.count_nonzero(usable)),
                                          int(chosen.shape[0]))
        target_rows = np.nonzero(usable)[0]
        return rows.with_forced_elements(
            target_rows, slot[usable] // 3, slot[usable] % 3)

    def _get_basis_surface(self, kind):
        """源侧基准空间的三角形 BVH。返回 (surface, domain, extent)。

        顶点域基准 → BVH 顶点池是逐顶点值,三角形用顶点索引(基准=位置时就是普通 3D 表面);
        角点域基准 → BVH 顶点池是逐角点值,三角形用 loop 索引(基准=UV 时就是 UV 空间),
        并剔除零面积退化三角形(未展开的面),避免所有查询被吸到同一点。
        """
        def build():
            source = self.source_snapshot
            values, domain, name, components = self._get_source_basis_values(kind)
            extent = self._value_extent(values)
            self._require_basis_extent(extent, kind, name, components, "Source")
            triangle_vertices = source.triangle_vertex_indices
            triangle_loops = source.triangle_loop_indices
            if triangle_vertices.shape[0] == 0:
                raise ConformError("Source mesh has no faces to match against")
            if domain == POINT:
                return (SurfaceCorrespondence(
                    values, triangle_vertices, triangle_vertices, triangle_loops),
                    domain, extent)
            corners = values[triangle_loops]
            doubled_area = np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1)
            keep = doubled_area > 1e-14
            if np.any(keep):
                # 有面撑得开面积 → 零面积的那些是"没展开的面",留着会把查询全吸过去。
                degenerate_count = int(np.count_nonzero(~keep))
                if degenerate_count:
                    self.warnings.append(
                        f"{degenerate_count:,} source faces have zero area in "
                        f"'{name}' and were left out of the matching space")
            else:
                # 一个面也撑不开:一维基准、或平面着色的法线(整面共用一个法线)。
                # 此时插值本就无意义,全留着让重心退化成"取值最接近的那个角",
                # 这仍是有效匹配 —— 真正致命的"整层塌成一点"已由 extent 闸拦下。
                keep = np.ones(doubled_area.shape[0], dtype=bool)
            return (SurfaceCorrespondence(
                values, triangle_loops[keep], triangle_vertices[keep],
                triangle_loops[keep]), domain, extent)
        return self._cached_match(("basis_surface", kind), build)

    def _basis_source_name(self, kind):
        """源侧基准实际用的层名(空 = 活动项)。"""
        return resolve_source_name(
            self.source_snapshot, kind, self.settings.match_basis_name_source) or ""

    def _get_target_basis_values(self, kind, domain):
        """目标侧基准值:顶点域基准取逐顶点值,角点域基准取逐角点值。

        目标层名的选法:显式指定 > 与源同名 > 目标的活动项 ——
        "两个网格的同一份数据"默认按名字对上,而不是各用各的活动层。
        """
        def build():
            target_name = resolve_match_name(
                self.target_snapshot, kind, self.settings.match_basis_name_target,
                self._basis_source_name(kind))
            values, target_domain, name, components = self._read_basis_values(
                self.target_snapshot, kind, target_name, is_target=True)
            self._require_basis_extent(
                self._value_extent(values), kind, name, components, "Target")
            if target_domain != domain:
                # 两侧同一通道却落在不同域(颜色属性可点可角),折算到源侧的域再匹配。
                bridge = SourceDomainBridge(
                    self.target_snapshot.loop_vertex_indices,
                    self.target_snapshot.vertex_count)
                values = bridge.to_domain(values, target_domain, domain)
            return values
        return self._cached_match(("target_basis", kind), build)

    def _warn_if_collapsed(self, original, result):
        """结果尺寸相对原尺寸暴缩 = "匹配空间没对上"的直接证据,必须当场喊出来。

        距离检测抓不到这一类:目标的 UV 若被打包进源 UV 的一个小角落,查询点全都
        落在源面"内部",命中距离为零,但采到的却是源上极小一块 —— 表现就是整个
        网格缩成一团。拿结果本身的包围盒比才抓得住。
        """
        if original.shape[0] < 4:
            return
        before = self._value_extent(original)
        after = self._value_extent(result)
        if before <= 1e-9 or after >= before * 0.02:
            return
        self.warnings.append(
            f"Result shrank to {after / before:.1%} of the target's size — the two "
            f"sides of the matching basis probably do not line up (check that both "
            f"point at the same layer)")

    def _warn_if_basis_far(self, kind, distances, valid, extent):
        """两侧基准明明是"同一份数据"就该几乎零距离命中。

        平均命中距离相对基准尺寸偏大 = 十有八九两边指的不是同一层
        (源用 UVMap、目标用了另一套光照 UV),这正是"目标被吸成一小团"的成因。
        """
        if extent <= 1e-12 or not np.any(valid):
            return
        mean_distance = float(np.mean(distances[valid]))
        if mean_distance <= extent * 0.05:
            return
        self.warnings.append(
            f"Target values sit {mean_distance / extent:.0%} of the basis size away "
            f"from the source {channel_label(kind).lower()} on average — check that "
            f"both sides really point at the same layer")

    def _corner_query_geometry(self, values=None):
        """(导向偏置查询点, 真实角点值),都在匹配空间。

        偏置:角点值向所属面的平均值挪一点点来决定命中哪个面,接缝两侧各落到正确一侧;
        重心权重再用真实角点值无钳计算,边界处线性外推不内缩。
        values 为空 = 用顶点位置(基准就是形状时的老路径,面均值即多边形中心)。
        """
        target = self.target_snapshot
        if values is None:
            corner_values = target.vertex_positions[target.loop_vertex_indices]
            face_means = target.corner_face_centers
            if self._is_world_space():
                corner_values = transform_points(corner_values, self._target_matrix)
                face_means = transform_points(face_means, self._target_matrix)
        else:
            corner_values = values
            face_means = self._face_mean_of_corners(corner_values)
        nudged = corner_values + (
            face_means - corner_values) * self.settings.corner_sampling_bias
        return nudged, corner_values

    def _face_mean_of_corners(self, corner_values):
        """每个角点所属面的平均值 (L, C)(位置通道时等于多边形中心)。"""
        target = self.target_snapshot
        face_indices = target.loop_polygon_indices
        face_count = len(target.mesh.polygons)
        counts = np.bincount(face_indices, minlength=face_count).astype(np.float64)
        counts[counts == 0.0] = 1.0
        sums = np.empty((face_count, corner_values.shape[1]), dtype=np.float64)
        for channel in range(corner_values.shape[1]):
            sums[:, channel] = np.bincount(
                face_indices, weights=corner_values[:, channel], minlength=face_count)
        return (sums / counts[:, None])[face_indices]

    # ==================== 顶点域对应(Blender 顶点映射全集) ====================

    def get_vertex_correspondence(self):
        if self._vertex_correspondence is not None:
            return self._vertex_correspondence
        mapping = self.vertex_mapping()
        target = self.target_snapshot
        if mapping == 'TOPOLOGY':
            correspondence = self._build_vertex_topology()
        elif mapping == 'UV':
            # Custom 里的传统 UV 映射 = 以 UV 通道为基准的插值匹配。
            correspondence = self._build_vertex_basis(UV)
        elif mapping == 'BASIS':
            basis = self.settings.match_basis
            correspondence = self._build_vertex_basis(
                basis, self._basis_is_nearest(basis))
        elif mapping == 'NEAREST':
            correspondence = self._build_vertex_nearest()
        elif mapping in {'EDGE_NEAREST', 'EDGEINTERP_NEAREST'}:
            correspondence = self._build_vertex_edge(mapping)
        elif mapping == 'POLY_NEAREST':
            correspondence = self._build_vertex_nearest_face_vertex()
        else:
            # POLYINTERP_NEAREST / POLYINTERP_VNORPROJ:面重心插值(最近点 / 法线投射)。
            correspondence = self._build_vertex_face_interpolated(mapping)
        unmatched = int(np.count_nonzero(~correspondence.valid))
        if unmatched:
            self.warnings.append(
                f"{unmatched:,} of {target.vertex_count:,} vertices found no match "
                f"— they keep their original data")
        self._vertex_correspondence = correspondence
        return correspondence

    def _build_vertex_topology(self):
        source_count = self.source_snapshot.vertex_count
        target_count = self.target_snapshot.vertex_count
        if source_count != target_count:
            raise ConformError(
                f"Topology mapping needs equal vertex counts "
                f"(source {source_count:,}, target {target_count:,})")
        return TopologyVertexCorrespondence(target_count, self.source_bridge)

    def _basis_is_nearest(self, kind):
        """基准通道的匹配方式;投射只对形状基准有意义,其余退回插值并说明。"""
        method = self.settings.match_method
        if method == 'PROJECTED' and kind != POSITION:
            self.warnings.append(
                "Projection only works with the shape basis — matched by "
                "interpolation instead")
            return False
        return method == 'NEAREST'

    def _build_vertex_basis(self, kind, nearest=False):
        """以任意通道为基准的顶点域对应。

        顶点域基准(位置/权重/点色…) → 逐目标顶点在基准空间里找最近点;
        角点域基准(UV/法线/角色…) → 逐目标角点查询后按逆距离权重收敛到顶点,
        同一顶点的多份角点值(接缝)因此必然收敛到唯一结果。
        """
        target = self.target_snapshot
        surface, domain, extent = self._get_basis_surface(kind)
        values = self._get_target_basis_values(kind, domain)
        max_distance = self._search_max_distance()

        if domain == POINT:
            triangle_indices, hit_positions, distances = surface.query_nearest(
                values, max_distance)
            # 顶点域权重用命中点 + 内钳:采样结果必须落在源面上。
            rows = surface.resolve(
                triangle_indices, hit_positions, distances, clamp_inside=True)
            self._warn_if_basis_far(kind, distances, rows.valid, extent)
            if nearest:
                rows = rows.as_nearest()
            rows = self._apply_exact_basis_matches(
                rows, surface, kind, domain, values)
            return DirectVertexCorrespondence(rows)

        if target.loop_count == 0:
            raise ConformError(
                f"Target mesh has no face corners to match by "
                f"{channel_label(kind).lower()}")
        first_indices, inverse_indices = deduplicate_queries(values)
        triangle_indices, hit_positions, distances = surface.query_nearest(
            values[first_indices], max_distance)
        rows = surface.resolve(
            triangle_indices, hit_positions, distances, clamp_inside=True)
        self._warn_if_basis_far(kind, distances, rows.valid, extent)
        if nearest:
            rows = rows.as_nearest()
        rows = self._apply_exact_basis_matches(
            rows.expand(inverse_indices), surface, kind, domain, values)
        return CombinedVertexCorrespondence(
            rows,
            target.loop_vertex_indices,
            target.vertex_count)

    def _build_vertex_nearest(self):
        """NEAREST = Nearest Vertex:最近源顶点 one-hot。"""
        kd_tree = self._get_source_vertex_kd()
        points = self._get_target_match_positions()
        indices, distances = query_kd_nearest(kd_tree, points)
        valid = indices >= 0
        safe_indices = np.where(valid, indices, 0)
        count = points.shape[0]
        return GatherCorrespondence(
            safe_indices[:, None], np.ones((count, 1), dtype=np.float64),
            distances, valid, POINT, self.source_bridge)

    def _build_vertex_edge(self, mapping):
        """EDGE_NEAREST = 最近边的较近端点;EDGEINTERP_NEAREST = 最近边上最近点线性插值。"""
        edge_query = self._get_edge_query()
        points = self._get_target_match_positions()
        edge_indices, hit_positions, distances = edge_query.query_nearest(
            points, self._search_max_distance())
        valid = edge_indices >= 0
        safe_edges = np.where(valid, edge_indices, 0)
        endpoints = edge_query.edge_vertex_indices[safe_edges]
        source_positions = self._get_source_match_positions()
        position_a = source_positions[endpoints[:, 0]]
        segment = source_positions[endpoints[:, 1]] - position_a
        segment_length_squared = np.einsum('ij,ij->i', segment, segment)
        parameter = np.einsum('ij,ij->i', hit_positions - position_a, segment)
        parameter = np.divide(
            parameter, segment_length_squared,
            out=np.zeros_like(parameter), where=segment_length_squared > 1e-24)
        np.clip(parameter, 0.0, 1.0, out=parameter)
        count = points.shape[0]
        if mapping == 'EDGE_NEAREST':
            chosen = np.where(parameter > 0.5, endpoints[:, 1], endpoints[:, 0])
            return GatherCorrespondence(
                chosen[:, None], np.ones((count, 1), dtype=np.float64), distances, valid,
                POINT, self.source_bridge)
        weights = np.stack((1.0 - parameter, parameter), axis=1)
        return GatherCorrespondence(
            endpoints, weights, distances, valid, POINT, self.source_bridge)

    def _build_vertex_nearest_face_vertex(self):
        """POLY_NEAREST = Nearest Face Vertex:最近面上离查询点最近的角顶点。"""
        surface = self._get_surface_3d()
        points = self._get_target_match_positions()
        triangle_indices, _hit_positions, distances = surface.query_nearest(
            points, self._search_max_distance())
        valid = triangle_indices >= 0
        safe_triangles = np.where(valid, triangle_indices, 0)
        polygons = self.source_snapshot.triangle_polygon_indices[safe_triangles]
        best_vertices = self._nearest_polygon_vertex(polygons, points)
        count = points.shape[0]
        return GatherCorrespondence(
            best_vertices[:, None], np.ones((count, 1), dtype=np.float64),
            distances, valid, POINT, self.source_bridge)

    def _nearest_polygon_vertex(self, polygons, query_points):
        """逐行求多边形角顶点中离查询点最近者(ragged argmin 全向量化)。"""
        source = self.source_snapshot
        counts = source.polygon_loop_totals[polygons]
        row_indices, within_offsets = ragged_arange(counts)
        candidate_loops = source.polygon_loop_starts[polygons][row_indices] + within_offsets
        candidate_vertices = source.loop_vertex_indices[candidate_loops]
        offsets = self._get_source_match_positions()[candidate_vertices] \
            - query_points[row_indices]
        scores = np.einsum('ij,ij->i', offsets, offsets)
        present_segments, best_rows = segment_best_rows(
            row_indices, scores, take_maximum=False)
        best_vertices = np.zeros(polygons.shape[0], dtype=np.int64)
        best_vertices[present_segments] = candidate_vertices[best_rows]
        return best_vertices

    def _build_vertex_face_interpolated(self, mapping):
        surface = self._get_surface_3d()
        points = self._get_target_match_positions()
        if mapping == 'POLYINTERP_VNORPROJ':
            directions = self._match_space_target_normals(
                self.target_snapshot.vertex_normals)
            triangle_indices, hit_positions, distances = surface.query_ray(
                points, directions,
                self.settings.project_max_distance or None,
                self.settings.project_bidirectional)
        else:
            triangle_indices, hit_positions, distances = surface.query_nearest(
                points, self._search_max_distance())
        rows = surface.resolve(
            triangle_indices, hit_positions, distances, clamp_inside=True)
        return DirectVertexCorrespondence(rows)

    # ==================== 角点域对应(Blender 角点映射全集) ====================

    def get_corner_correspondence(self):
        if self._corner_correspondence is not None:
            return self._corner_correspondence
        mapping = self.corner_mapping()
        target = self.target_snapshot
        if mapping == 'TOPOLOGY':
            correspondence = self._build_corner_topology()
        elif mapping == 'UV':
            correspondence = self._build_corner_basis(UV)
        elif mapping == 'BASIS':
            basis = self.settings.match_basis
            correspondence = self._build_corner_basis(
                basis, self._basis_is_nearest(basis))
        elif mapping in {'NEAREST_NORMAL', 'NEAREST_POLYNOR'}:
            correspondence = self._build_corner_nearest_normal(mapping)
        elif mapping == 'NEAREST_POLY':
            correspondence = self._build_corner_nearest_face_corner()
        else:
            # POLYINTERP_NEAREST / POLYINTERP_LNORPROJ:面角插值(最近点 / 角法线投射)。
            correspondence = self._build_corner_face_interpolated(mapping)
        unmatched = int(np.count_nonzero(~correspondence.valid))
        if unmatched:
            self.warnings.append(
                f"{unmatched:,} of {target.loop_count:,} face corners found no match "
                f"— they keep their original data")
        self._corner_correspondence = correspondence
        return correspondence

    def _build_corner_topology(self):
        source_count = self.source_snapshot.loop_count
        target_count = self.target_snapshot.loop_count
        if source_count != target_count:
            raise ConformError(
                f"Topology mapping needs equal corner counts "
                f"(source {source_count:,}, target {target_count:,})")
        return TopologyCornerCorrespondence(target_count, self.source_bridge)

    def _build_corner_basis(self, kind, nearest=False):
        """以任意通道为基准的角点域对应。

        角点域基准 → 用真实查询值做无钳权重,命中三角形线性延拓,层间恒等传输精确;
        顶点域基准 → 角点取所属顶点的基准值,导向偏置决定命中面(接缝两侧各归各的面)。
        """
        target = self.target_snapshot
        surface, domain, extent = self._get_basis_surface(kind)
        values = self._get_target_basis_values(kind, domain)
        max_distance = self._search_max_distance()

        if domain == POINT:
            corner_values = values[target.loop_vertex_indices]
            nudged, exact_values = self._corner_query_geometry(corner_values)
            triangle_indices, _hit_positions, distances = surface.query_nearest(
                nudged, max_distance)
            rows = surface.resolve(
                triangle_indices, exact_values, distances, clamp_inside=False)
            self._warn_if_basis_far(kind, distances, rows.valid, extent)
            rows = self._apply_exact_basis_matches(
                rows, surface, kind, domain, corner_values)
        else:
            first_indices, inverse_indices = deduplicate_queries(values)
            queries = values[first_indices]
            triangle_indices, _hit_positions, distances = surface.query_nearest(
                queries, max_distance)
            rows = surface.resolve(
                triangle_indices, queries, distances, clamp_inside=False)
            self._warn_if_basis_far(kind, distances, rows.valid, extent)
            rows = self._apply_exact_basis_matches(
                rows.expand(inverse_indices), surface, kind, domain, values)
        if nearest:
            rows = rows.as_nearest()
        return DirectCornerCorrespondence(rows)

    def _build_corner_nearest_normal(self, mapping):
        """NEAREST_NORMAL = 最近源顶点上"拆分法线最匹配"的角点;
        NEAREST_POLYNOR = 最近源顶点上"所属面法线最匹配"的角点。"""
        source = self.source_snapshot
        target = self.target_snapshot
        if source.loop_count == 0:
            raise ConformError("Source mesh has no face corners")
        kd_tree = self._get_source_vertex_kd()
        vertex_points = self._get_target_match_positions()
        nearest_source_vertex, vertex_distances = query_kd_nearest(kd_tree, vertex_points)
        vertex_valid = nearest_source_vertex >= 0
        loop_vertex = target.loop_vertex_indices
        corner_source_vertex = np.where(vertex_valid, nearest_source_vertex, 0)[loop_vertex]

        sorted_loops, offsets = source.vertex_loop_csr
        counts = (offsets[1:] - offsets[:-1])[corner_source_vertex]
        row_indices, within_offsets = ragged_arange(counts)
        candidate_loops = sorted_loops[
            offsets[corner_source_vertex][row_indices] + within_offsets]

        if mapping == 'NEAREST_NORMAL':
            target_normals = self._match_space_target_normals(target.corner_normals)
            source_normals = self._match_space_source_normals(source.corner_normals)
            scores = np.einsum(
                'ij,ij->i', source_normals[candidate_loops], target_normals[row_indices])
        else:
            target_normals = self._match_space_target_normals(
                target.polygon_normals[target.loop_polygon_indices])
            source_face_normals = self._match_space_source_normals(source.polygon_normals)
            scores = np.einsum(
                'ij,ij->i',
                source_face_normals[source.loop_polygon_indices[candidate_loops]],
                target_normals[row_indices])

        present_segments, best_rows = segment_best_rows(
            row_indices, scores, take_maximum=True)
        loop_count = target.loop_count
        best_loops = np.zeros(loop_count, dtype=np.int64)
        found = np.zeros(loop_count, dtype=bool)
        best_loops[present_segments] = candidate_loops[best_rows]
        found[present_segments] = True
        valid = found & vertex_valid[loop_vertex]
        distances = vertex_distances[loop_vertex]
        return GatherCorrespondence(
            best_loops[:, None], np.ones((loop_count, 1), dtype=np.float64),
            distances, valid, CORNER, self.source_bridge)

    def _build_corner_nearest_face_corner(self):
        """NEAREST_POLY = Nearest Corner of Nearest Face:最近面上离角点最近的角。"""
        source = self.source_snapshot
        target = self.target_snapshot
        surface = self._get_surface_3d()
        corner_points = target.vertex_positions[target.loop_vertex_indices]
        if self._is_world_space():
            corner_points = transform_points(corner_points, self._target_matrix)
        triangle_indices, _hit_positions, distances = surface.query_nearest(
            corner_points, self._search_max_distance())
        valid = triangle_indices >= 0
        safe_triangles = np.where(valid, triangle_indices, 0)
        polygons = source.triangle_polygon_indices[safe_triangles]

        counts = source.polygon_loop_totals[polygons]
        row_indices, within_offsets = ragged_arange(counts)
        candidate_loops = source.polygon_loop_starts[polygons][row_indices] + within_offsets
        candidate_positions = self._get_source_match_positions()[
            source.loop_vertex_indices[candidate_loops]]
        offsets = candidate_positions - corner_points[row_indices]
        scores = np.einsum('ij,ij->i', offsets, offsets)
        present_segments, best_rows = segment_best_rows(
            row_indices, scores, take_maximum=False)
        loop_count = target.loop_count
        best_loops = np.zeros(loop_count, dtype=np.int64)
        best_loops[present_segments] = candidate_loops[best_rows]
        return GatherCorrespondence(
            best_loops[:, None], np.ones((loop_count, 1), dtype=np.float64),
            distances, valid, CORNER, self.source_bridge)

    def _build_corner_face_interpolated(self, mapping):
        surface = self._get_surface_3d()
        nudged, corner_positions = self._corner_query_geometry()
        if mapping == 'POLYINTERP_LNORPROJ':
            # Blender 语义:沿角点"拆分法线"投射(LNORPROJ = loop normal projected)。
            directions = self._match_space_target_normals(
                self.target_snapshot.corner_normals)
            triangle_indices, _hit_positions, distances = surface.query_ray(
                nudged, directions,
                self.settings.project_max_distance or None,
                self.settings.project_bidirectional)
        else:
            triangle_indices, _hit_positions, distances = surface.query_nearest(
                nudged, self._search_max_distance())
        rows = surface.resolve(
            triangle_indices, corner_positions, distances, clamp_inside=False)
        return DirectCornerCorrespondence(rows)

    # ==================== 影响权重管线 ====================

    def _get_vertex_influence_base(self):
        """mix × 顶点组遮罩 × 选择遮罩(不含命中有效性与距离衰减)。"""
        if self._vertex_influence_base is not None:
            return self._vertex_influence_base
        settings = self.settings
        target = self.target_snapshot
        base = np.full(target.vertex_count, settings.mix_factor, dtype=np.float64)
        mask_name = settings.vertex_group_mask
        if mask_name:
            group_names = target.vertex_group_names
            if mask_name in group_names:
                weights = target.vertex_group_weight_matrix[:, group_names.index(mask_name)]
                if settings.invert_vertex_group_mask:
                    weights = 1.0 - weights
                base = base * weights
            else:
                self.warnings.append(
                    f"Mask vertex group '{mask_name}' not found on "
                    f"'{self.target_object.name}' — mask ignored")
        if settings.use_selection_only:
            base = base * target.vertex_selection.astype(np.float64)
        self._vertex_influence_base = base
        return base

    def _distance_falloff(self, distances):
        settings = self.settings
        if not settings.use_max_distance:
            return None
        if settings.distance_falloff > 0.0:
            # 从 max_distance - falloff 处开始线性衰减到 0。
            return np.clip(
                (settings.max_distance - distances) / settings.distance_falloff, 0.0, 1.0)
        return (distances <= settings.max_distance).astype(np.float64)

    def _vertex_influence(self, correspondence):
        cached = self._influence_cache.get(id(correspondence))
        if cached is not None:
            return cached
        influence = self._get_vertex_influence_base() * correspondence.valid
        falloff = self._distance_falloff(correspondence.distances)
        if falloff is not None:
            influence = influence * falloff
        self._influence_cache[id(correspondence)] = influence
        return influence

    def _corner_influence(self, correspondence):
        cached = self._influence_cache.get(id(correspondence))
        if cached is not None:
            return cached
        loop_vertex_indices = self.target_snapshot.loop_vertex_indices
        influence = self._get_vertex_influence_base()[loop_vertex_indices] * correspondence.valid
        falloff = self._distance_falloff(correspondence.distances)
        if falloff is not None:
            influence = influence * falloff
        self._influence_cache[id(correspondence)] = influence
        return influence

    # ==================== 数据传输 ====================

    def transfer_shape(self):
        settings = self.settings
        correspondence = self.get_vertex_correspondence()
        influence = self._vertex_influence(correspondence)[:, None]
        source_positions = self.source_snapshot.vertex_positions
        sampled = correspondence.sample(source_positions, POINT)
        if settings.snap_shape_to_vertices:
            sampled = snap_positions_to_nearest(
                sampled, source_positions, correspondence.valid)
        mapped = transform_points(sampled, self._position_matrix)
        original = self.target_snapshot.vertex_positions
        self._warn_if_collapsed(original[correspondence.valid],
                                mapped[correspondence.valid])

        if settings.shape_as_shape_key:
            current = original
            result = current + (mapped - current) * influence
            mesh = self.target_object.data
            if mesh.shape_keys is None:
                self.target_object.shape_key_add(name="Basis", from_mix=False)
            basis_positions = read_shape_key_positions(
                mesh.shape_keys.key_blocks[0], self.target_snapshot.vertex_count)
            key_block = add_numbered_shape_key(
                self.target_object, f"{self.source_object.name}.Conformed")
            # 相对形态键是叠加的:存"相对当前可见形状的修正量",加上去正好得到目标形状,
            # 已有形态键的值一个都不用动,再应用一次就在这一层之上继续推进。
            write_shape_key_positions(key_block, basis_positions + (result - current))
            key_block.value = 1.0
            self.target_object.data.update()
            return f"Shape (shape key '{key_block.name}')"

        result = original + (mapped - original) * influence
        apply_vertex_positions(self.target_object, result)
        return "Shape"

    def transfer_vertex_groups(self):
        settings = self.settings
        source = self.source_snapshot
        names = source.vertex_group_names
        locks = source.vertex_group_locks
        if settings.vertex_groups_exclude_locked:
            kept_indices = [index for index, locked in enumerate(locks) if not locked]
        else:
            kept_indices = list(range(len(names)))
        if not kept_indices:
            self.warnings.append("Source has no vertex groups to transfer")
            return None
        correspondence = self.get_vertex_correspondence()
        influence = self._vertex_influence(correspondence)
        sampled = correspondence.sample(
            source.vertex_group_weight_matrix[:, kept_indices], POINT)
        target = self.target_snapshot
        existing_names = target.vertex_group_names
        existing_matrix = target.vertex_group_weight_matrix
        for column, source_index in enumerate(kept_indices):
            group_name = names[source_index]
            if group_name in existing_names:
                existing = existing_matrix[:, existing_names.index(group_name)]
            else:
                existing = np.zeros(target.vertex_count, dtype=np.float64)
            blended = existing + (sampled[:, column] - existing) * influence
            write_vertex_group_weights(self.target_object, group_name, blended)
        return f"Vertex Groups ({len(kept_indices)})"

    def _resolve_uv_target_name(self, source_layer_name, transferring_all):
        settings = self.settings
        if settings.uv_write_mode == 'ACTIVE' and not transferring_all:
            active = self.target_object.data.uv_layers.active
            if active is not None:
                return active.name, False
        if settings.uv_write_mode == 'NEW':
            return source_layer_name, True
        return source_layer_name, False

    def transfer_uv_layers(self):
        settings = self.settings
        source = self.source_snapshot
        target_mesh = self.target_object.data
        if self.target_snapshot.loop_count == 0:
            self.warnings.append("Target mesh has no face corners — UVs skipped")
            return None
        if settings.uv_transfer_all:
            layer_names = source.uv_layer_names
        else:
            chosen = settings.uv_transfer_layer_source or source.active_uv_layer_name
            layer_names = [chosen] if chosen else []
        if not layer_names:
            self.warnings.append("Source has no UV layers to transfer")
            return None
        correspondence = self.get_corner_correspondence()
        influence = self._corner_influence(correspondence)[:, None]
        written_names = []
        for layer_name in layer_names:
            source_uv = source.read_uv_layer(layer_name)
            if source_uv is None:
                continue
            sampled = correspondence.sample(source_uv, CORNER)
            target_name, force_new = self._resolve_uv_target_name(
                layer_name, settings.uv_transfer_all)
            if force_new:
                actual_name = add_numbered_uv_layer(target_mesh, target_name)
            else:
                actual_name = ensure_uv_layer(target_mesh, target_name)
            if actual_name is None:
                self.warnings.append(
                    f"UV layer limit (8) reached — '{layer_name}' skipped")
                continue
            existing = read_uv_layer(target_mesh, actual_name)
            blended = existing + (sampled - existing) * influence
            write_uv_layer(target_mesh, actual_name, blended)
            written_names.append(actual_name)
        if not written_names:
            return None
        return f"UVs ({len(written_names)})"

    def transfer_color_attributes(self):
        settings = self.settings
        source = self.source_snapshot
        target_mesh = self.target_object.data
        if settings.color_transfer_all:
            attribute_names = source.color_attribute_names
        else:
            chosen = settings.color_transfer_attribute or source.active_color_attribute_name
            attribute_names = [chosen] if chosen else []
        if not attribute_names:
            self.warnings.append("Source has no color attributes to transfer")
            return None
        transferred_count = 0
        for attribute_name in attribute_names:
            payload = source.read_color_attribute(attribute_name)
            if payload is None:
                continue
            domain, data_type, values = payload
            if domain == 'CORNER':
                if self.target_snapshot.loop_count == 0:
                    self.warnings.append(
                        f"Target has no face corners — color attribute "
                        f"'{attribute_name}' skipped")
                    continue
                correspondence = self.get_corner_correspondence()
                influence = self._corner_influence(correspondence)[:, None]
            elif domain == 'POINT':
                correspondence = self.get_vertex_correspondence()
                influence = self._vertex_influence(correspondence)[:, None]
            else:
                self.warnings.append(
                    f"Color attribute '{attribute_name}' uses unsupported domain "
                    f"'{domain}' — skipped")
                continue
            sampled = correspondence.sample(values, domain)
            attribute, recreated = ensure_color_attribute(
                target_mesh, attribute_name, data_type, domain)
            if recreated:
                self.warnings.append(
                    f"Color attribute '{attribute_name}' was recreated to match "
                    f"the source domain/type")
            # 用创建后的真实名字回读:与同名泛型属性撞名时 Blender 会自动改名。
            existing = read_color_attribute(target_mesh, attribute.name)[2]
            blended = existing + (sampled - existing) * influence
            if data_type == 'BYTE_COLOR':
                np.clip(blended, 0.0, 1.0, out=blended)
            write_color_attribute(attribute, blended)
            transferred_count += 1
        if transferred_count == 0:
            return None
        if target_mesh.color_attributes.active_color_index < 0:
            target_mesh.color_attributes.active_color_index = 0
        return f"Colors ({transferred_count})"

    def transfer_corner_normals(self):
        target = self.target_snapshot
        if target.loop_count == 0:
            self.warnings.append("Target mesh has no face corners — normals skipped")
            return None
        correspondence = self.get_corner_correspondence()
        influence = self._corner_influence(correspondence)[:, None]
        sampled = correspondence.sample(self.source_snapshot.corner_normals, CORNER)
        if self._is_world_space():
            linear = self._position_matrix[:3, :3]
            try:
                inverse_transpose = np.linalg.inv(linear).T
            except np.linalg.LinAlgError:
                raise ConformError("Source matrix is not invertible (zero scale?)")
            sampled = transform_directions(sampled, inverse_transpose)
        existing = target.corner_normals
        sampled = normalized_rows(sampled, fallback=existing)
        blended = normalized_rows(
            existing + (sampled - existing) * influence, fallback=existing)
        write_corner_normals(self.target_object.data, blended)
        return "Custom Normals"

    def _capture_evaluated_source_vertex_positions(self):
        """抓取源对象当前求值结果的顶点坐标(形态键隔离快照用,一次性求值网格)。"""
        depsgraph = self._context.evaluated_depsgraph_get()
        evaluated_object = self.source_object.evaluated_get(depsgraph)
        evaluated_mesh = evaluated_object.to_mesh()
        try:
            count = len(evaluated_mesh.vertices)
            buffer = np.empty(count * 3, dtype=np.float32)
            evaluated_mesh.vertices.foreach_get("co", buffer)
            return buffer.astype(np.float64).reshape(count, 3)
        finally:
            evaluated_object.to_mesh_clear()

    def transfer_shape_keys(self):
        settings = self.settings
        source_shape_keys = self.source_object.data.shape_keys
        if source_shape_keys is None or len(source_shape_keys.key_blocks) < 2:
            self.warnings.append("Source has no shape keys to transfer")
            return None
        correspondence = self.get_vertex_correspondence()
        influence = self._vertex_influence(correspondence)[:, None]
        key_blocks = source_shape_keys.key_blocks
        use_evaluated = settings.use_evaluated_source
        source_vertex_count = self.source_snapshot.vertex_count
        linear = self._position_matrix[:3, :3]

        target_mesh = self.target_object.data
        if target_mesh.shape_keys is None:
            self.target_object.shape_key_add(name="Basis", from_mix=False)
        target_vertex_count = self.target_snapshot.vertex_count
        target_basis_positions = read_shape_key_positions(
            target_mesh.shape_keys.key_blocks[0], target_vertex_count)

        transferred_count = 0
        value_backup = None
        if use_evaluated:
            value_backup = [key_block.value for key_block in key_blocks]
            for key_block in key_blocks:
                key_block.value = 0.0
        try:
            if use_evaluated:
                # 全零值求值 = 干净的形变基准(修复旧版拿"当前混合值"当基准的偏差)。
                base_positions = self._capture_evaluated_source_vertex_positions()
                if base_positions.shape[0] != source_vertex_count:
                    self.warnings.append(
                        "Evaluated source vertex count changed between snapshots "
                        "— shape keys skipped")
                    return None
            else:
                base_positions = read_shape_key_positions(
                    key_blocks[0], source_vertex_count)
            base_sampled = correspondence.sample(base_positions, POINT)
            for key_block in list(key_blocks)[1:]:
                if settings.shape_keys_exclude_muted and key_block.mute:
                    continue
                if use_evaluated:
                    key_block.value = 1.0
                    key_positions = self._capture_evaluated_source_vertex_positions()
                    key_block.value = 0.0
                    if key_positions.shape[0] != source_vertex_count:
                        self.warnings.append(
                            f"Evaluated vertex count changed at shape key "
                            f"'{key_block.name}' — remaining keys skipped")
                        break
                else:
                    key_positions = read_shape_key_positions(
                        key_block, source_vertex_count)
                key_sampled = correspondence.sample(key_positions, POINT)
                if settings.snap_shape_keys_to_vertices:
                    key_sampled = snap_positions_to_nearest(
                        key_sampled, key_positions, correspondence.valid)
                # 形变增量只经线性部分映射(平移对增量无意义)。
                delta = (key_sampled - base_sampled) @ linear.T
                target_key, _created = ensure_shape_key(
                    self.target_object, key_block.name)
                existing_positions = read_shape_key_positions(
                    target_key, target_vertex_count)
                existing_delta = existing_positions - target_basis_positions
                final_delta = existing_delta + (delta - existing_delta) * influence
                write_shape_key_positions(
                    target_key, target_basis_positions + final_delta)
                try:
                    # min/max/min 三段赋值:绕开 RNA 把 min 钳在旧 max 之下的顺序陷阱。
                    target_key.slider_min = key_block.slider_min
                    target_key.slider_max = key_block.slider_max
                    target_key.slider_min = key_block.slider_min
                except (AttributeError, TypeError):
                    pass
                transferred_count += 1
        finally:
            if value_backup is not None:
                for key_block, value in zip(key_blocks, value_backup):
                    key_block.value = value
        self.target_object.data.update()
        if transferred_count == 0:
            self.warnings.append("No shape keys were transferred")
            return None
        summary = f"Shape Keys ({transferred_count})"
        if settings.shape_keys_transfer_drivers:
            driver_count = transfer_shape_key_drivers(
                self.source_object, self.target_object,
                settings.armature_source, settings.armature_target)
            summary += f" + {driver_count} driver(s)"
        return summary

    # ==================== 通道转换(任意通道 → 任意通道) ====================

    def _match_space_source_positions(self, positions):
        if self._is_world_space():
            return transform_points(positions, self._source_matrix)
        return positions

    def _match_space_to_target_positions(self, positions):
        return transform_points(positions, self._match_to_target_matrix)

    def _match_space_to_target_normals(self, normals):
        """匹配空间方向 → 目标局部方向(法线的逆转置的逆 = 线性部分的转置)。"""
        if not self._is_world_space():
            return normals
        return normalized_rows(
            transform_directions(normals, self._target_matrix[:3, :3].T))

    def _warn_on_corner_fold(self, corner_values):
        """角点域 → 顶点域会取均值:接缝/硬边上的顶点因此被"拉平",必须说清楚。"""
        bridge = self.source_bridge
        averaged = bridge.corner_to_vertex(corner_values)
        deviation = np.abs(
            corner_values - averaged[bridge.loop_vertex_indices]).max(axis=1)
        split_loops = deviation > 1e-6
        if not np.any(split_loops):
            return
        split_vertices = int(np.count_nonzero(np.bincount(
            bridge.loop_vertex_indices, weights=split_loops.astype(np.float64),
            minlength=bridge.vertex_count) > 0.0))
        self.warnings.append(
            f"{split_vertices:,} vertices carry more than one value (UV seams / hard "
            f"edges) — those were averaged into a single vertex value")

    def _convert_component_picks(self):
        settings = self.settings
        if settings.convert_component_mode != 'CUSTOM':
            return None
        return [settings.convert_component_x, settings.convert_component_y,
                settings.convert_component_z, settings.convert_component_w]

    def convert_channel(self):
        """执行转换。按名字成批时逐条跑同一条管线,对应关系与影响权重只算一次。"""
        settings = self.settings
        source_kind = settings.convert_from
        target_kind = settings.convert_to
        if settings.convert_all_named and source_kind in NAMED_CHANNELS:
            names = channel_names(self.source_snapshot, source_kind)
            if not names:
                raise ConformError(
                    f"Source has no {channel_label(source_kind).lower()} to convert")
            converted = []
            for name in names:
                # 成批时目标名恒等于源名 —— 骨骼/顶点组、形态键都按名字一一对上。
                converted.append(self._convert_one(source_kind, name,
                                                   target_kind, name))
            return (f"{channel_label(source_kind)} × {len(converted)} → "
                    f"{channel_label(target_kind)} (matched by name)")
        summary = self._convert_one(
            source_kind, settings.convert_from_name,
            target_kind, settings.convert_to_name)
        return summary

    def _convert_one(self, source_kind, source_name, target_kind, target_name):
        """把源的任意一个通道转成目标的任意一个通道。

        管线:读源 → (位置/方向进匹配空间) → 对应关系采样(可跨域)→ 数值重映射 →
        分量重映射 → (位置/方向回目标空间)→ 按影响权重与现值混合 → 写回。
        """
        settings = self.settings
        try:
            source_data = read_source_channel(
                self.source_snapshot, source_kind, source_name)
        except ChannelError as error:
            raise ConformError(str(error))

        values = source_data.values
        if source_data.positional:
            values = self._match_space_source_positions(values)
        elif source_data.directional:
            values = self._match_space_source_normals(values)

        resolved_target_name = target_name or default_target_name(
            source_data.name, target_kind)
        try:
            target = TargetChannel(
                self.target_object, target_kind, resolved_target_name,
                source_data.domain, source_data.components,
                settings.convert_to_domain, settings.convert_to_attribute_type)
        except ChannelError as error:
            raise ConformError(str(error))

        if target.domain == POINT:
            correspondence = self.get_vertex_correspondence()
            influence = self._vertex_influence(correspondence)[:, None]
            if source_data.domain == CORNER:
                self._warn_on_corner_fold(values)
        else:
            correspondence = self.get_corner_correspondence()
            influence = self._corner_influence(correspondence)[:, None]

        sampled = correspondence.sample(values, source_data.domain)
        sampled = remap_values(
            sampled, settings.convert_value_remap,
            settings.convert_value_scale, settings.convert_value_offset)
        result = adapt_components(
            sampled, target.components, self._convert_component_picks())

        if target_kind in POSITIONAL_CHANNELS:
            # 数值一律按"所选空间"解释:世界模式下 UV / 颜色等非位置数据也被当成
            # 世界坐标写回,所以 UV → 形态键就是把 UV 布局摊在世界的 1×1 方格里。
            result = self._match_space_to_target_positions(result)
            if source_data.positional:
                # 位置转位置才谈得上"塌缩";UV→顶点这类本来就该变形状,不适用。
                valid = correspondence.valid
                self._warn_if_collapsed(
                    self.target_snapshot.vertex_positions[valid], result[valid])
        elif target_kind in DIRECTIONAL_CHANNELS:
            result = self._match_space_to_target_normals(result)

        existing = target.read_existing(self.target_snapshot)
        blended = existing + (result - existing) * influence
        try:
            target.write(blended)
        except ChannelError as error:
            raise ConformError(str(error))
        target.focus()

        if target.recreated:
            self.warnings.append(
                f"'{target.name}' was recreated to fit the converted data")
        return (f"{channel_label(source_kind)} '{source_data.name}' → "
                f"{channel_label(target_kind)} '{target.name}'")

    # ==================== 执行入口 ====================

    def run(self):
        if self.settings.mode == 'CONVERT':
            summary = self.convert_channel()
            self.summaries.append(summary)
            return self.summaries, self.warnings

        settings = self.settings
        transfer_plan = []
        if settings.use_shape:
            transfer_plan.append(self.transfer_shape)
        if settings.use_vertex_groups:
            transfer_plan.append(self.transfer_vertex_groups)
        if settings.use_uv_layers:
            transfer_plan.append(self.transfer_uv_layers)
        if settings.use_color_attributes:
            transfer_plan.append(self.transfer_color_attributes)
        if settings.use_corner_normals:
            transfer_plan.append(self.transfer_corner_normals)
        if settings.use_shape_keys:
            # 形态键必须最后跑:求值隔离快照会刷新 depsgraph,使源快照的
            # 求值网格指针失效,之后不得再懒加载任何源数据。
            transfer_plan.append(self.transfer_shape_keys)
        if not transfer_plan:
            raise ConformError("Enable at least one data type to conform")

        # 依赖目标几何的映射:在 Shape 写回之前预建对应,冻结几何快照。
        needs_vertex = (settings.use_shape or settings.use_shape_keys
                        or settings.use_vertex_groups or settings.use_color_attributes)
        needs_corner = (settings.use_uv_layers or settings.use_color_attributes
                        or settings.use_corner_normals)
        if needs_vertex and self.vertex_mapping() not in _POSITION_INDEPENDENT_MAPPINGS:
            self.get_vertex_correspondence()
        if (needs_corner and self.corner_mapping() not in _POSITION_INDEPENDENT_MAPPINGS
                and self.target_snapshot.loop_count > 0):
            self.get_corner_correspondence()

        for transfer in transfer_plan:
            summary = transfer()
            if summary:
                self.summaries.append(summary)
        if not self.summaries:
            raise ConformError("Nothing was transferred — see warnings for details")
        return self.summaries, self.warnings

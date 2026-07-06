# 表面贴合会话:把设置解析成对应关系与影响权重,驱动全部数据类型的传输。
# 架构:一次会话只构建一次顶点域/角点域对应关系,全部数据类型复用同一内核采样
# (旧版每种数据各跑一遍投射,本版 O(1) 次对应 + 每类一次向量化采样)。
# 影响权重统一管线:mix × 顶点组遮罩 × 选择遮罩 × 距离衰减 × 命中有效性,
# 所有数据按 result = existing + (sampled - existing) × influence 混合落地。

import numpy as np

from .correspondence import (
    SurfaceCorrespondence,
    DirectVertexCorrespondence,
    CombinedVertexCorrespondence,
    TopologyVertexCorrespondence,
    DirectCornerCorrespondence,
    TopologyCornerCorrespondence,
    deduplicate_uv_queries,
    snap_positions_to_nearest,
)
from .mesh_buffers import (
    MeshBufferSnapshot,
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
    write_vertex_positions,
    ensure_shape_key,
    read_shape_key_positions,
    write_shape_key_positions,
    write_corner_normals,
    write_vertex_group_weights,
)
from .shape_key_drivers import transfer_shape_key_drivers


class ConformError(RuntimeError):
    """配置或数据不满足传输前提时抛出,由算子层转成 report。"""


class ConformSession:
    def __init__(self, context, settings, target_object):
        self._context = context
        self.settings = settings
        self.target_object = target_object

        source_object = settings.source_object
        if source_object is None or source_object.type != 'MESH':
            raise ConformError("Pick a mesh object as the source")
        if source_object == target_object:
            raise ConformError("Source and target are the same object")
        self.source_object = source_object

        depsgraph = context.evaluated_depsgraph_get() if settings.use_evaluated_source else None
        self.source_snapshot = MeshBufferSnapshot(
            source_object, settings.use_evaluated_source, depsgraph)
        self.target_snapshot = MeshBufferSnapshot(target_object)

        if self.target_snapshot.vertex_count == 0:
            raise ConformError("Target mesh has no vertices")
        if settings.matching_domain != 'TOPOLOGY' and len(self.source_snapshot.mesh.polygons) == 0:
            raise ConformError("Source mesh has no faces to sample")

        self._source_matrix = matrix_to_numpy(source_object.matrix_world)
        self._target_matrix = matrix_to_numpy(target_object.matrix_world)
        if settings.transform_space == 'WORLD':
            try:
                target_inverse = np.linalg.inv(self._target_matrix)
            except np.linalg.LinAlgError:
                raise ConformError("Target matrix is not invertible (zero scale?)")
            # 位置数据映射:源局部 → 世界 → 目标局部。
            self._position_matrix = target_inverse @ self._source_matrix
        else:
            self._position_matrix = np.identity(4, dtype=np.float64)

        self._surface_3d = None
        self._uv_surface = None
        self._vertex_correspondence = None
        self._corner_correspondence = None
        self._vertex_influence_base = None
        self._influence_cache = {}
        self.warnings = []
        self.summaries = []

    def free(self):
        self.source_snapshot.free()
        self.target_snapshot.free()

    # ==================== 对应关系构建 ====================

    def _is_world_space(self):
        return self.settings.transform_space == 'WORLD'

    def _search_max_distance(self):
        return self.settings.max_distance if self.settings.use_max_distance else None

    def _get_surface_3d(self):
        """源表面 3D BVH。WORLD 模式在世界空间建树,距离阈值即世界单位。"""
        if self._surface_3d is None:
            source = self.source_snapshot
            positions = source.vertex_positions
            if self._is_world_space():
                positions = transform_points(positions, self._source_matrix)
            self._surface_3d = SurfaceCorrespondence(
                positions,
                source.triangle_vertex_indices,
                source.triangle_vertex_indices,
                source.triangle_loop_indices)
        return self._surface_3d

    def _get_uv_surface(self):
        """源 UV 空间 BVH:逐角点 UV 升维到 z=0,三角形即 loop 三角形。
        剔除零面积退化三角形(未展开面),避免查询被吸到 (0, 0)。"""
        if self._uv_surface is None:
            source = self.source_snapshot
            layer_name = self.settings.uv_match_layer_source or source.active_uv_layer_name
            if not layer_name:
                raise ConformError("Source needs a UV layer for UV matching")
            uv_coordinates = source.read_uv_layer(layer_name)
            if uv_coordinates is None:
                raise ConformError(
                    f"Source UV layer '{layer_name}' not found for UV matching")
            positions = np.zeros((uv_coordinates.shape[0], 3), dtype=np.float64)
            positions[:, :2] = uv_coordinates
            triangle_loops = source.triangle_loop_indices
            corner_uv = uv_coordinates[triangle_loops]
            edge_one = corner_uv[:, 1] - corner_uv[:, 0]
            edge_two = corner_uv[:, 2] - corner_uv[:, 0]
            doubled_area = np.abs(
                edge_one[:, 0] * edge_two[:, 1] - edge_one[:, 1] * edge_two[:, 0])
            keep = doubled_area > 1e-14
            if not np.any(keep):
                raise ConformError(
                    f"Source UV layer '{layer_name}' is fully degenerate (zero area)")
            self._uv_surface = SurfaceCorrespondence(
                positions,
                triangle_loops[keep],
                source.triangle_vertex_indices[keep],
                triangle_loops[keep])
        return self._uv_surface

    def _target_vertex_query_points(self):
        positions = self.target_snapshot.vertex_positions
        if self._is_world_space():
            positions = transform_points(positions, self._target_matrix)
        return positions

    def _target_vertex_query_normals(self):
        normals = self.target_snapshot.vertex_normals
        if self._is_world_space():
            # 法线按位置矩阵的逆转置变换(非均匀缩放安全)。
            inverse_transpose = np.linalg.inv(self._target_matrix[:3, :3]).T
            normals = normalized_rows(transform_directions(normals, inverse_transpose))
        return normals

    def _read_target_match_uv(self):
        target = self.target_snapshot
        layer_name = self.settings.uv_match_layer_target or target.active_uv_layer_name
        if not layer_name:
            raise ConformError("Target needs a UV layer for UV matching")
        uv_coordinates = target.read_uv_layer(layer_name)
        if uv_coordinates is None:
            raise ConformError(f"Target UV layer '{layer_name}' not found for UV matching")
        return uv_coordinates

    def get_vertex_correspondence(self):
        """顶点域对应(形状/形态键/顶点组/点域颜色共用)。"""
        if self._vertex_correspondence is not None:
            return self._vertex_correspondence
        settings = self.settings
        target = self.target_snapshot
        domain = settings.matching_domain
        if domain == 'TOPOLOGY':
            source_count = self.source_snapshot.vertex_count
            if source_count != target.vertex_count:
                raise ConformError(
                    f"Topology matching needs equal vertex counts "
                    f"(source {source_count:,}, target {target.vertex_count:,})")
            correspondence = TopologyVertexCorrespondence(target.vertex_count)
        elif domain == 'UV':
            if target.loop_count == 0:
                raise ConformError("Target mesh has no face corners for UV matching")
            target_uv = self._read_target_match_uv()
            first_indices, inverse_indices = deduplicate_uv_queries(target_uv)
            queries = np.zeros((first_indices.shape[0], 3), dtype=np.float64)
            queries[:, :2] = target_uv[first_indices]
            surface = self._get_uv_surface()
            triangle_indices, hit_positions, distances = surface.query_nearest(
                queries, self._search_max_distance())
            # 顶点域权重用命中点 + 内钳:采样结果必须落在源面上。
            rows = surface.resolve(
                triangle_indices, hit_positions, distances, clamp_inside=True)
            correspondence = CombinedVertexCorrespondence(
                rows.expand(inverse_indices),
                target.loop_vertex_indices,
                target.vertex_count)
        else:
            surface = self._get_surface_3d()
            points = self._target_vertex_query_points()
            if settings.surface_method == 'PROJECT':
                directions = self._target_vertex_query_normals()
                triangle_indices, hit_positions, distances = surface.query_ray(
                    points, directions,
                    self.settings.project_max_distance or None,
                    settings.project_bidirectional)
            else:
                triangle_indices, hit_positions, distances = surface.query_nearest(
                    points, self._search_max_distance())
            rows = surface.resolve(
                triangle_indices, hit_positions, distances, clamp_inside=True)
            correspondence = DirectVertexCorrespondence(rows)
        unmatched = int(np.count_nonzero(~correspondence.valid))
        if unmatched:
            self.warnings.append(
                f"{unmatched:,} of {target.vertex_count:,} vertices found no match "
                f"— they keep their original data")
        self._vertex_correspondence = correspondence
        return correspondence

    def get_corner_correspondence(self):
        """角点域对应(UV/角域颜色/自定义法线共用)。

        3D 匹配用"导向偏置"查询:角点向所属面中心偏移一点决定命中面
        (接缝两侧的角点各落到正确一侧),重心权重再用真实角点无钳计算
        (边界处线性外推,UV 不内缩)。
        """
        if self._corner_correspondence is not None:
            return self._corner_correspondence
        settings = self.settings
        target = self.target_snapshot
        domain = settings.matching_domain
        if domain == 'TOPOLOGY':
            source_count = self.source_snapshot.loop_count
            if source_count != target.loop_count:
                raise ConformError(
                    f"Topology matching needs equal corner counts "
                    f"(source {source_count:,}, target {target.loop_count:,})")
            correspondence = TopologyCornerCorrespondence(target.loop_count)
        elif domain == 'UV':
            target_uv = self._read_target_match_uv()
            first_indices, inverse_indices = deduplicate_uv_queries(target_uv)
            queries = np.zeros((first_indices.shape[0], 3), dtype=np.float64)
            queries[:, :2] = target_uv[first_indices]
            surface = self._get_uv_surface()
            triangle_indices, _hit_positions, distances = surface.query_nearest(
                queries, self._search_max_distance())
            rows = surface.resolve(
                triangle_indices, queries, distances, clamp_inside=False)
            correspondence = DirectCornerCorrespondence(rows.expand(inverse_indices))
        else:
            loop_vertex_indices = target.loop_vertex_indices
            corner_positions = target.vertex_positions[loop_vertex_indices]
            bias = settings.corner_sampling_bias
            nudged = corner_positions + (
                target.corner_face_centers - corner_positions) * bias
            if self._is_world_space():
                nudged = transform_points(nudged, self._target_matrix)
                corner_positions = transform_points(corner_positions, self._target_matrix)
            surface = self._get_surface_3d()
            if settings.surface_method == 'PROJECT':
                directions = self._target_vertex_query_normals()[loop_vertex_indices]
                triangle_indices, _hit_positions, distances = surface.query_ray(
                    nudged, directions,
                    self.settings.project_max_distance or None,
                    settings.project_bidirectional)
            else:
                triangle_indices, _hit_positions, distances = surface.query_nearest(
                    nudged, self._search_max_distance())
            rows = surface.resolve(
                triangle_indices, corner_positions, distances, clamp_inside=False)
            correspondence = DirectCornerCorrespondence(rows)
        unmatched = int(np.count_nonzero(~correspondence.valid))
        if unmatched:
            self.warnings.append(
                f"{unmatched:,} of {target.loop_count:,} face corners found no match "
                f"— they keep their original data")
        self._corner_correspondence = correspondence
        return correspondence

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
                    f"Mask vertex group '{mask_name}' not found on target — mask ignored")
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
        sampled = correspondence.sample(source_positions)
        if settings.snap_shape_to_vertices:
            sampled = snap_positions_to_nearest(
                sampled, source_positions, correspondence.valid)
        mapped = transform_points(sampled, self._position_matrix)
        original = self.target_snapshot.vertex_positions

        if settings.shape_as_shape_key:
            key_block, _created = ensure_shape_key(
                self.target_object, f"{self.source_object.name}.Conformed")
            vertex_count = self.target_snapshot.vertex_count
            base_for_blend = read_shape_key_positions(key_block, vertex_count)
            result = base_for_blend + (mapped - base_for_blend) * influence
            write_shape_key_positions(key_block, result)
            key_block.value = 1.0
            self.target_object.data.update()
            return "Shape (as shape key)"

        result = original + (mapped - original) * influence
        target_mesh = self.target_object.data
        if target_mesh.shape_keys is not None:
            # 目标带形态键时:Basis 与全部键整体平移同样的位移,保住各键的相对形变
            # (只写网格顶点在有键时视口不生效,是旧版的隐性缺陷)。
            key_blocks = target_mesh.shape_keys.key_blocks
            vertex_count = self.target_snapshot.vertex_count
            basis_positions = read_shape_key_positions(key_blocks[0], vertex_count)
            shift = result - basis_positions
            for key_block in key_blocks:
                key_positions = read_shape_key_positions(key_block, vertex_count)
                write_shape_key_positions(key_block, key_positions + shift)
        write_vertex_positions(target_mesh, result)
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
        sampled = correspondence.sample(source.vertex_group_weight_matrix[:, kept_indices])
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
            sampled = correspondence.sample(source_uv)
            target_name, force_new = self._resolve_uv_target_name(
                layer_name, settings.uv_transfer_all)
            actual_name = ensure_uv_layer(target_mesh, target_name, force_new=force_new)
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
            sampled = correspondence.sample(values)
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
        sampled = correspondence.sample(self.source_snapshot.corner_normals)
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
            base_sampled = correspondence.sample(base_positions)
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
                key_sampled = correspondence.sample(key_positions)
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

    # ==================== 执行入口 ====================

    def run(self):
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

        if settings.matching_domain == 'SURFACE':
            # 3D 匹配依赖目标顶点坐标:在 Shape 写回之前预建对应,冻结几何快照。
            needs_vertex = (settings.use_shape or settings.use_shape_keys
                            or settings.use_vertex_groups or settings.use_color_attributes)
            needs_corner = (settings.use_uv_layers or settings.use_color_attributes
                            or settings.use_corner_normals)
            if needs_vertex:
                self.get_vertex_correspondence()
            if needs_corner and self.target_snapshot.loop_count > 0:
                self.get_corner_correspondence()

        for transfer in transfer_plan:
            summary = transfer()
            if summary:
                self.summaries.append(summary)
        if not self.summaries:
            raise ConformError("Nothing was transferred — see warnings for details")
        return self.summaries, self.warnings

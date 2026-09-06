# 表面对应内核:BVH/KD 查询 + 重心/聚合插值采样。
# 纯数学层:只依赖 numpy 与 mathutils,不 import bpy。
# 设计要点:
#   1. 对应关系统一表达成「每个目标元素 → K 个源元素索引 + 权重」:
#      K=1 单点吸附(Nearest 系),K=2 边插值,K=3 面重心插值 —— 一个采样内核吃全部映射。
#   2. 同一套内核同时服务 3D 空间(BVH 顶点池 = 网格顶点)与 UV 空间
#      (BVH 顶点池 = 逐角点 UV 升维到 z=0),靠 triangle_vertex_indices /
#      triangle_corner_indices 双索引把命中三角形还原到源网格的顶点域与角点域。
#   3. UV 空间的顶点域采样必须经 CombinedVertexCorrespondence 收敛:
#      同一网格顶点的多个 UV loop 按逆距离权重合并成唯一结果,
#      修复旧版"最后写入者赢"导致的合并顶点/接缝顶点错乱。
#   4. 每种对应关系都实现统一签名 sample(data, domain):源数据是顶点域还是角点域
#      由调用方声明,与对应关系自身的域无关 —— 这是"任意通道 → 任意通道"互转的地基
#      (跨域时经 SourceDomainBridge 折算,角点→顶点取均值,顶点→角点直接展开)。

import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

# 域标识符,与 Blender attribute domain 一致。
POINT = 'POINT'
CORNER = 'CORNER'

# find_nearest / ray_cast 的"无限"搜索半径(mathutils 默认上限量级)。
_UNLIMITED_DISTANCE = 1.0e18
# 逆距离权重的抗除零项。
_DISTANCE_EPSILON = 1e-9


def compute_barycentric_weights(points, triangle_corners, clamp_inside):
    """逐行计算 points 相对 triangle_corners 的重心权重。

    points:           (N, 3) float64,查询点(可离开三角形平面,等价于沿法线投影)
    triangle_corners: (N, 3, 3) float64,每行一个三角形的三个角点
    clamp_inside:     True  = 钳回三角形内部并重归一(顶点域采样,结果必须落在面上)
                      False = 允许有限外推(角点域采样,保证 UV/法线在面边界处不内缩),
                              权重截断到 [-1, 2] 后重归一,防止病态外推爆炸
    退化三角形回退为最近角点的 one-hot 权重。
    """
    corner_a = triangle_corners[:, 0]
    edge_ab = triangle_corners[:, 1] - corner_a
    edge_ac = triangle_corners[:, 2] - corner_a
    to_point = points - corner_a

    dot_ab_ab = np.einsum('ij,ij->i', edge_ab, edge_ab)
    dot_ab_ac = np.einsum('ij,ij->i', edge_ab, edge_ac)
    dot_ac_ac = np.einsum('ij,ij->i', edge_ac, edge_ac)
    dot_point_ab = np.einsum('ij,ij->i', to_point, edge_ab)
    dot_point_ac = np.einsum('ij,ij->i', to_point, edge_ac)

    denominator = dot_ab_ab * dot_ac_ac - dot_ab_ac * dot_ab_ac
    # 相对阈值判退化,保证尺度不变性;1e-30 兜底纯零三角形。
    degenerate = denominator <= np.maximum(dot_ab_ab * dot_ac_ac, 1e-30) * 1e-12
    safe_denominator = np.where(degenerate, 1.0, denominator)

    weights = np.empty(points.shape, dtype=np.float64)
    weights[:, 1] = (dot_ac_ac * dot_point_ab - dot_ab_ac * dot_point_ac) / safe_denominator
    weights[:, 2] = (dot_ab_ab * dot_point_ac - dot_ab_ac * dot_point_ab) / safe_denominator
    weights[:, 0] = 1.0 - weights[:, 1] - weights[:, 2]

    if clamp_inside:
        np.clip(weights, 0.0, None, out=weights)
        weight_sum = weights.sum(axis=1, keepdims=True)
        np.divide(weights, weight_sum, out=weights, where=weight_sum > 1e-20)
    else:
        np.clip(weights, -1.0, 2.0, out=weights)
        weight_sum = weights.sum(axis=1, keepdims=True)
        np.divide(weights, weight_sum, out=weights, where=np.abs(weight_sum) > 1e-12)

    if np.any(degenerate):
        rows = np.nonzero(degenerate)[0]
        offsets = triangle_corners[rows] - points[rows][:, None, :]
        corner_distances = np.einsum('nkj,nkj->nk', offsets, offsets)
        nearest_corner = np.argmin(corner_distances, axis=1)
        weights[rows] = 0.0
        weights[rows, nearest_corner] = 1.0
    return weights


# ==================== ragged(变长段)向量化工具 ====================

def ragged_arange(counts):
    """counts (N,) → (row_indices, within_offsets):
    row_indices 标记每个展开元素属于哪一行,within_offsets 是行内 [0..counts[i]) 序号。"""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    row_indices = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    if total == 0:
        return row_indices, np.empty(0, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    within_offsets = np.arange(total, dtype=np.int64) - starts[row_indices]
    return row_indices, within_offsets


def segment_best_rows(segment_indices, scores, take_maximum):
    """每段取分数最优的一行。返回 (出现过的段 id, 对应的最优行号)。
    实现:lexsort 按 (段, 分数) 排序后取每段首行,全程向量化。"""
    if segment_indices.shape[0] == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    keys = -scores if take_maximum else scores
    order = np.lexsort((keys, segment_indices))
    sorted_segments = segment_indices[order]
    first_mask = np.empty(sorted_segments.shape[0], dtype=bool)
    first_mask[0] = True
    first_mask[1:] = sorted_segments[1:] != sorted_segments[:-1]
    return sorted_segments[first_mask], order[first_mask]


# ==================== 源域折算 ====================

class SourceDomainBridge:
    """源网格的 角点域 ⇄ 顶点域 折算器。

    让任何一种对应关系都能采样任何一个域的源数据:
    角点 → 顶点取同顶点各角点均值(接缝/硬边的多值在此收敛),
    顶点 → 角点按 loop→vertex 表直接展开(无损)。
    """

    __slots__ = ("loop_vertex_indices", "vertex_count", "_loop_counts")

    def __init__(self, loop_vertex_indices, vertex_count):
        self.loop_vertex_indices = np.asarray(loop_vertex_indices, dtype=np.int64)
        self.vertex_count = int(vertex_count)
        counts = np.bincount(
            self.loop_vertex_indices, minlength=self.vertex_count).astype(np.float64)
        # 孤立顶点(无角点)计数置 1 防除零,其行恒为 0。
        self._loop_counts = np.where(counts > 0.0, counts, 1.0)

    def corner_to_vertex(self, corner_data):
        data = np.asarray(corner_data, dtype=np.float64)
        channel_count = data.shape[1]
        result = np.empty((self.vertex_count, channel_count), dtype=np.float64)
        for channel in range(channel_count):
            result[:, channel] = np.bincount(
                self.loop_vertex_indices, weights=data[:, channel],
                minlength=self.vertex_count)
        return result / self._loop_counts[:, None]

    def vertex_to_corner(self, vertex_data):
        return np.asarray(vertex_data, dtype=np.float64)[self.loop_vertex_indices]

    def to_domain(self, data, from_domain, to_domain):
        if from_domain == to_domain:
            return np.asarray(data, dtype=np.float64)
        if to_domain == POINT:
            return self.corner_to_vertex(data)
        return self.vertex_to_corner(data)


# ==================== 对应关系表达 ====================

class GatherCorrespondence:
    """通用聚合对应:每行 K 个源元素索引 + 权重。

    索引指向 index_domain 域的源元素:
    K=1 = 单元素吸附(Nearest Vertex / Nearest Corner 系映射),
    K=2 = 边端点插值,K=3 及以上 = 面插值。
    采样另一个域的源数据时先经 bridge 折算,再走同一套 gather。
    """

    __slots__ = ("valid", "distances", "index_domain", "_indices", "_weights", "_bridge")

    def __init__(self, indices, weights, distances, valid, index_domain, bridge):
        self._indices = indices      # (N, K) int64
        self._weights = weights      # (N, K) float64
        self.distances = distances   # (N,) float64
        self.valid = valid           # (N,) bool
        self.index_domain = index_domain
        self._bridge = bridge

    def sample(self, data, domain=POINT):
        data = self._bridge.to_domain(data, domain, self.index_domain)
        return np.einsum('nkc,nk->nc', data[self._indices], self._weights)

    def dominant_corners(self):
        rows = np.arange(self._weights.shape[0], dtype=np.int64)
        return self._indices[rows, np.argmax(self._weights, axis=1)]


class CorrespondenceRows:
    """一次三角形命中查询的逐行结果:命中三角形 + 重心权重,可对任意源数据插值采样。"""

    __slots__ = ("valid", "distances", "_triangle_indices", "_weights", "_owner")

    def __init__(self, valid, triangle_indices, weights, distances, owner):
        self.valid = valid                       # (N,) bool
        self.distances = distances               # (N,) float64,未命中为 inf
        self._triangle_indices = triangle_indices  # (N,) int64,未命中行已替换为 0(安全 gather)
        self._weights = weights                  # (N, 3) float64
        self._owner = owner

    def expand(self, inverse_indices):
        """按去重逆映射把"唯一查询"的结果展开回原始行序(零拷贝语义的 gather)。"""
        return CorrespondenceRows(
            self.valid[inverse_indices],
            self._triangle_indices[inverse_indices],
            self._weights[inverse_indices],
            self.distances[inverse_indices],
            self._owner,
        )

    def sample_vertex_data(self, vertex_data):
        """按顶点域源数据插值。vertex_data: (V_source, C) → (N, C)。"""
        data = np.asarray(vertex_data, dtype=np.float64)
        gather = self._owner.triangle_vertex_indices[self._triangle_indices]
        return np.einsum('nkc,nk->nc', data[gather], self._weights)

    def sample_corner_data(self, corner_data):
        """按角点域(loop)源数据插值。corner_data: (L_source, C) → (N, C)。
        三角形内部的角点数据插值永不跨越 UV 接缝/法线硬边,这是逐面采样的核心正确性。"""
        data = np.asarray(corner_data, dtype=np.float64)
        gather = self._owner.triangle_corner_indices[self._triangle_indices]
        return np.einsum('nkc,nk->nc', data[gather], self._weights)

    def with_forced_elements(self, target_rows, triangle_indices, corner_slots):
        """指定行改成直取某个源三角形的某个角(one-hot 权重),不做任何插值。

        基准值与源逐位相同的元素就该原样取那一处 —— 命中距离归零,权重不再参与插值。
        """
        weights = self._weights.copy()
        weights[target_rows] = 0.0
        weights[target_rows, corner_slots] = 1.0
        triangles = self._triangle_indices.copy()
        triangles[target_rows] = triangle_indices
        valid = self.valid.copy()
        valid[target_rows] = True
        distances = self.distances.copy()
        distances[target_rows] = 0.0
        return CorrespondenceRows(valid, triangles, weights, distances, self._owner)

    def dominant_corners(self):
        """每个目标角点权重最大的那个源角点(loop)索引。"""
        rows = np.arange(self._weights.shape[0], dtype=np.int64)
        slots = np.argmax(self._weights, axis=1)
        return self._owner.triangle_corner_indices[self._triangle_indices][rows, slots]

    def as_nearest(self):
        """把重心权重塌成命中三角形里最近那个角的 one-hot。

        基准通道的"最近元素"语义:不插值,直接取值最接近的那个源元素,
        数值一个不改地搬过来(离散数据/整数 ID 类通道要的就是这个)。
        """
        rows = np.arange(self._weights.shape[0], dtype=np.int64)
        weights = np.zeros_like(self._weights)
        weights[rows, np.argmax(self._weights, axis=1)] = 1.0
        return CorrespondenceRows(
            self.valid, self._triangle_indices, weights, self.distances, self._owner)


class SurfaceCorrespondence:
    """源表面的三角形查询结构:一次构建,多次查询/采样。"""

    __slots__ = ("triangle_count", "triangle_vertex_indices", "triangle_corner_indices",
                 "_triangle_corner_positions", "_bvh_tree", "_element_slots")

    def __init__(self, bvh_positions, bvh_triangles, triangle_vertex_indices, triangle_corner_indices):
        """
        bvh_positions:           (P, 3) float64,BVH 顶点池(3D=网格顶点,UV=逐角点 UV 升维)
        bvh_triangles:           (T, 3) int64,索引 bvh_positions 的三角形
        triangle_vertex_indices: (T, 3) int64,三角形三个角对应的源网格顶点索引
        triangle_corner_indices: (T, 3) int64,三角形三个角对应的源 loop 索引
        """
        self.triangle_count = int(bvh_triangles.shape[0])
        self.triangle_vertex_indices = triangle_vertex_indices
        self.triangle_corner_indices = triangle_corner_indices
        # 预 gather 角点坐标,重心计算全程向量化。
        self._triangle_corner_positions = bvh_positions[bvh_triangles]
        self._bvh_tree = BVHTree.FromPolygons(
            bvh_positions.tolist(), bvh_triangles.tolist(), all_triangles=True)
        self._element_slots = {}

    def settle_by_orientation(self, query_points, reference_normals, triangle_normals,
                              triangle_indices, distances, tie_radii):
        """把命中到"背对着的那一层"的行改判到朝向一致的同距候选上。

        贴在一起的双层布料(两张面重合、法线相反、各用一座 UV 岛)在位置上完全分不开,
        命中哪一张纯看 BVH 遍历顺序,必然采错一半;朝向才是能分开它们的判据。
        只有朝向真的相反的行才重查,几何正常的网格一行都不会走这条路;
        候选也只收"同距"的那些,隔着距离的另一片(头发卡片这种)绝不会被抢过去。
        """
        valid = triangle_indices >= 0
        agreement = np.einsum(
            'ij,ij->i', triangle_normals[np.where(valid, triangle_indices, 0)],
            reference_normals)
        ambiguous = np.nonzero(valid & (agreement < 0.0))[0]
        if ambiguous.shape[0] == 0:
            return triangle_indices, distances
        settled_indices = triangle_indices.copy()
        settled_distances = distances.copy()
        find_range = self._bvh_tree.find_nearest_range
        for row in ambiguous.tolist():
            reference = reference_normals[row]
            best_index = -1
            best_distance = 0.0
            best_agreement = 0.0
            for _location, _normal, index, distance in find_range(
                    query_points[row].tolist(), distances[row] + tie_radii[row]):
                candidate = float(np.dot(triangle_normals[index], reference))
                if candidate <= 0.0:
                    continue
                if best_index < 0 or distance < best_distance or (
                        distance == best_distance and candidate > best_agreement):
                    best_index, best_distance, best_agreement = index, distance, candidate
            if best_index >= 0:
                settled_indices[row] = best_index
                settled_distances[row] = best_distance
        return settled_indices, settled_distances

    def element_slots(self, domain, element_count):
        """源元素 → 含它的某个角在三角形数组里的扁平位置(slot//3 = 三角形, slot%3 = 角)。

        没有被任何三角形用到的元素(如整面零面积被剔除)返回 -1,调用方据此退回几何查询。
        """
        key = (domain, element_count)
        if key not in self._element_slots:
            table = (self.triangle_corner_indices if domain == CORNER
                     else self.triangle_vertex_indices)
            slots = np.full(element_count, -1, dtype=np.int64)
            flat = table.ravel()
            slots[flat] = np.arange(flat.shape[0], dtype=np.int64)
            self._element_slots[key] = slots
        return self._element_slots[key]

    def query_nearest(self, query_points, max_distance=None):
        """最近表面点查询。返回 (triangle_indices, hit_positions, distances),未命中行 index=-1。"""
        count = query_points.shape[0]
        triangle_indices = np.full(count, -1, dtype=np.int64)
        hit_positions = np.zeros((count, 3), dtype=np.float64)
        distances = np.full(count, np.inf, dtype=np.float64)
        search_radius = float(max_distance) if max_distance is not None else _UNLIMITED_DISTANCE
        find_nearest = self._bvh_tree.find_nearest
        # 唯一的逐点 Python 循环:mathutils 无批量查询 API,循环体保持最小。
        for index, point in enumerate(query_points.tolist()):
            location, _normal, triangle_index, distance = find_nearest(point, search_radius)
            if triangle_index is not None:
                triangle_indices[index] = triangle_index
                hit_positions[index] = location
                distances[index] = distance
        return triangle_indices, hit_positions, distances

    def query_ray(self, query_points, directions, ray_distance=None, bidirectional=True):
        """沿方向射线投射查询。bidirectional=True 时双向投射取更近命中
        (优于旧版"正向优先":避免正向命中远表面却忽略背向近表面)。"""
        count = query_points.shape[0]
        triangle_indices = np.full(count, -1, dtype=np.int64)
        hit_positions = np.zeros((count, 3), dtype=np.float64)
        distances = np.full(count, np.inf, dtype=np.float64)
        limit = float(ray_distance) if ray_distance else _UNLIMITED_DISTANCE
        ray_cast = self._bvh_tree.ray_cast
        for index, (point, direction) in enumerate(zip(query_points.tolist(), directions.tolist())):
            location, _normal, triangle_index, distance = ray_cast(point, direction, limit)
            if bidirectional:
                backward = ray_cast(
                    point, (-direction[0], -direction[1], -direction[2]), limit)
                if backward[2] is not None and (triangle_index is None or backward[3] < distance):
                    location, _normal, triangle_index, distance = backward
            if triangle_index is not None:
                triangle_indices[index] = triangle_index
                hit_positions[index] = location
                distances[index] = distance
        return triangle_indices, hit_positions, distances

    def resolve(self, triangle_indices, weight_points, distances, clamp_inside):
        """把原始命中结果解析成可采样的 CorrespondenceRows。

        weight_points: 用于计算重心权重的点。
            顶点域传命中点(权重必然在三角形内);
            角点域传"真实角点"(配合导向偏置查询实现接缝正确)。
        """
        valid = triangle_indices >= 0
        safe_indices = np.where(valid, triangle_indices, 0)
        corners = self._triangle_corner_positions[safe_indices]
        weights = compute_barycentric_weights(weight_points, corners, clamp_inside)
        return CorrespondenceRows(valid, safe_indices, weights, distances, self)

    def resolve_at_surface(self, triangle_indices, corner_points, distances,
                           open_boundary_edges):
        """角点域解析:一律取三角形上的最近点,只有源表面真的到头的地方才外推。

        横向外推是有代价的:目标角点一旦不是恰好落在源面上(抽面出来的 LOD、
        不同拓扑的变体),拿它相对三角形的重心去线性延拓会把 UV 甩到图集外面 ——
        实测抽面到 0.9 时最远甩到 3.82,而正确取值全在 0..1 内。
        真正需要延拓的只有一种情形:源表面在这里就结束了(开放边界),目标却还往外
        伸出去一点。那时钳死会把整圈边界压成一条零面积的带子,延拓才是对的。
        所以判据是"最近点落在的那条边是不是源的开放边界",而不是一个外推幅度上限。
        """
        valid = triangle_indices >= 0
        safe_indices = np.where(valid, triangle_indices, 0)
        corners = self._triangle_corner_positions[safe_indices]

        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        unit = np.divide(normals, lengths, out=np.zeros_like(normals),
                         where=lengths > 1e-20)
        elevation = np.einsum('ij,ij->i', corner_points - corners[:, 0], unit)
        projected = corner_points - unit * elevation[:, None]

        clamped = compute_barycentric_weights(projected, corners, True)
        # 被钳掉的那几个分量为零,说明最近点落在"正对着它们"的那条边上。
        on_edge = clamped <= 1e-12
        outward_allowed = on_edge & open_boundary_edges[safe_indices]
        if not np.any(outward_allowed):
            return CorrespondenceRows(valid, safe_indices, clamped, distances, self)

        # 延拓只保留"垂直于边界边往外"的那一份:沿着边界方向的位置仍取钳住的结果。
        # 目标角点在源面上横向挪开(抽面出来的 LOD 必然如此)属于沿边方向,
        # 让它参与线性延拓就会把取值甩到图集外面去 —— 实测最远甩到 3.82。
        free = compute_barycentric_weights(corner_points, corners, False)
        outward = np.where(outward_allowed, np.minimum(free, 0.0), 0.0)
        weights = clamped * (1.0 - outward.sum(axis=1))[:, None] + outward
        return CorrespondenceRows(valid, safe_indices, weights, distances, self)


# ==================== 源岛(角点通道的连通分量) ====================

def connected_component_labels(left_nodes, right_nodes, node_count):
    """连通分量标号,返回逐节点的代表元(每个分量取其中最小的节点号)。

    经典的"取邻居最小 + 指针跳跃"迭代,全程 numpy:边按起点排一次序,之后每一轮
    只是一次 gather 加一次 reduceat。并查集那种逐边 Python 循环在几万条边上要几百毫秒,
    这里几轮就收敛,而且没有 ufunc.at 那类慢路径。
    """
    labels = np.arange(node_count, dtype=np.int64)
    if left_nodes.shape[0] == 0:
        return labels
    sources = np.concatenate((left_nodes, right_nodes))
    targets = np.concatenate((right_nodes, left_nodes))
    order = np.argsort(sources, kind='stable')
    sources = sources[order]
    targets = targets[order]
    counts = np.bincount(sources, minlength=node_count)
    present = np.nonzero(counts > 0)[0]
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))[present]
    every_node_has_an_edge = present.shape[0] == node_count

    while True:
        neighbour_minimum = np.minimum.reduceat(labels[targets], starts)
        if every_node_has_an_edge:
            candidate = np.minimum(labels, neighbour_minimum)
        else:
            candidate = labels.copy()
            candidate[present] = np.minimum(candidate[present], neighbour_minimum)
        while True:
            jumped = candidate[candidate]
            if np.array_equal(jumped, candidate):
                break
            candidate = jumped
        if np.array_equal(candidate, labels):
            return labels
        labels = candidate


def corner_island_labels(loop_starts, loop_totals, loop_vertex_indices, values):
    """角点通道的"岛"标号:同一张面的角点连通,同一顶点上取值相同的角点也连通。

    UV 的接缝就是这张连通图上被剪开的地方,所以岛是**逐层**的 —— 每层 UV 有自己的接缝,
    颜色属性有自己的断点。拿一层的岛去锁另一层必然错。
    """
    loop_count = loop_vertex_indices.shape[0]
    if loop_count == 0:
        return np.empty(0, dtype=np.int64)
    face_of_loop = np.repeat(
        np.arange(loop_starts.shape[0], dtype=np.int64), loop_totals)
    within_face_left = loop_starts[face_of_loop]
    within_face_right = np.arange(loop_count, dtype=np.int64)

    magnitude = float(np.max(np.abs(values))) if values.size else 1.0
    step = 16777216.0 / max(magnitude, 1.0)
    quantized = np.round(values * step).astype(np.int64)
    # 只需要知道"同一顶点上的哪几个角点取值相同",不需要给取值编全局号:
    # 按 (顶点, 取值) 排一次序比相邻行即可,省掉 unique(axis=0) 那条慢路径。
    keys = [quantized[:, column] for column in range(quantized.shape[1] - 1, -1, -1)]
    order = np.lexsort(tuple(keys) + (loop_vertex_indices,))
    sorted_values = quantized[order]
    same = ((loop_vertex_indices[order][1:] == loop_vertex_indices[order][:-1])
            & (sorted_values[1:] == sorted_values[:-1]).all(axis=1))
    shared_left = order[:-1][same]
    shared_right = order[1:][same]

    return connected_component_labels(
        np.concatenate((within_face_left, shared_left)),
        np.concatenate((within_face_right, shared_right)),
        loop_count)


def lock_faces_to_islands(island_of_source_corner, source_corner_positions,
                          target_corner_positions, target_face_of_corner,
                          chosen_source_corners, valid):
    """把每张目标面整张锁进同一座源岛,返回改判后的"每个目标角点取自哪个源角点"。

    一张目标面是一块连通的曲面,它的像也必须连通。角点各自独立解算时,一张骑在源接缝上
    的面会让几个角落到接缝两侧,UV 上就是一条横穿整张图集的长边 —— 模型上看就是一片
    乱掉的三角形。这里按"整张面贴哪座岛最紧"(各角点到该岛最近点的距离之和最小)选定
    一座岛,再把不在这座岛上的角点改判到该岛离它最近的角点。
    只改真的骑在两座岛上的面,别的面一个角点都不动。
    """
    result = np.array(chosen_source_corners, dtype=np.int64, copy=True)
    usable = np.nonzero(valid)[0]
    if usable.shape[0] == 0:
        return result, 0
    islands_per_corner = island_of_source_corner[chosen_source_corners]
    faces = target_face_of_corner[usable]
    face_count = int(target_face_of_corner.max()) + 1

    # 哪些面骑在两座岛上:按 (面, 岛) 排一次序数不同的组即可,全程向量化,
    # 不必为了做这个判断把七千多张面逐张走一遍 Python。
    order = np.lexsort((islands_per_corner[usable], faces))
    sorted_faces = faces[order]
    sorted_islands = islands_per_corner[usable][order]
    group_start = np.empty(order.shape[0], dtype=bool)
    group_start[0] = True
    group_start[1:] = ((sorted_faces[1:] != sorted_faces[:-1])
                       | (sorted_islands[1:] != sorted_islands[:-1]))
    distinct_islands = np.bincount(sorted_faces[group_start], minlength=face_count)
    straddling = np.nonzero(distinct_islands > 1)[0]
    if straddling.shape[0] == 0:
        return result, 0

    corner_order = usable[np.argsort(faces, kind='stable')]
    corners_per_face = np.bincount(faces, minlength=face_count)
    face_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(corners_per_face)))

    island_trees = {}
    island_members = {}

    def tree_for(island):
        if island not in island_trees:
            members = np.nonzero(island_of_source_corner == island)[0]
            tree = KDTree(members.shape[0])
            insert = tree.insert
            for slot, member in enumerate(members.tolist()):
                insert(source_corner_positions[member].tolist(), slot)
            tree.balance()
            island_members[island] = members
            island_trees[island] = tree
        return island_trees[island]

    locked_faces = 0
    for face in straddling.tolist():
        corners = corner_order[face_offsets[face]:face_offsets[face + 1]]
        candidates = np.unique(islands_per_corner[corners])
        locked_faces += 1
        best_island = -1
        best_cost = None
        best_choice = None
        for island in candidates.tolist():
            tree = tree_for(island)
            members = island_members[island]
            cost = 0.0
            choice = []
            for corner in corners.tolist():
                _location, slot, distance = tree.find(
                    target_corner_positions[corner].tolist())
                if slot is None:
                    cost = None
                    break
                cost += distance
                choice.append(members[slot])
            if cost is None:
                continue
            if best_cost is None or cost < best_cost:
                best_cost, best_island, best_choice = cost, island, choice
        if best_island < 0:
            continue
        for corner, member in zip(corners.tolist(), best_choice):
            if islands_per_corner[corner] != best_island:
                result[corner] = member
    return result, locked_faces
def build_kd_tree(positions):
    """mathutils KDTree 构建(最近顶点系映射用)。"""
    kd_tree = KDTree(positions.shape[0])
    insert = kd_tree.insert
    for index, coordinate in enumerate(positions.tolist()):
        insert(coordinate, index)
    kd_tree.balance()
    return kd_tree


def query_kd_nearest(kd_tree, query_points):
    """逐点最近邻查询。返回 (indices, distances),未命中(空树)行 index=-1。"""
    count = query_points.shape[0]
    indices = np.full(count, -1, dtype=np.int64)
    distances = np.full(count, np.inf, dtype=np.float64)
    find = kd_tree.find
    for index, point in enumerate(query_points.tolist()):
        _location, reference_index, distance = find(point)
        if reference_index is not None:
            indices[index] = reference_index
            distances[index] = distance
    return indices, distances


class DirectVertexCorrespondence:
    """3D 空间逐顶点直查的顶点域对应:一顶点一命中,天然一致。

    命中的是源三角形 + 重心权重,所以顶点域与角点域源数据都能在命中点精确插值。
    """

    __slots__ = ("valid", "distances", "_rows")

    def __init__(self, rows):
        self._rows = rows
        self.valid = rows.valid
        self.distances = rows.distances

    def sample(self, data, domain=POINT):
        if domain == CORNER:
            return self._rows.sample_corner_data(data)
        return self._rows.sample_vertex_data(data)


class CombinedVertexCorrespondence:
    """UV 空间的顶点域对应:逐 loop 查询后按逆距离权重收敛到顶点。

    核心保证:同一网格顶点的所有 UV loop(接缝/合并顶点的多重 UV)最终合并出
    唯一一份采样结果——顶点只会被放到一个位置,彻底修复旧版逐 loop 覆写的错乱。
    逆距离加权让"落在正确 UV 岛上的 loop"(命中距离≈0)天然主导,
    错误岛屿上的远命中权重趋零。
    """

    __slots__ = ("valid", "distances", "_loop_rows", "_loop_weights",
                 "_loop_vertex_indices", "_vertex_count")

    def __init__(self, loop_rows, loop_vertex_indices, vertex_count):
        inverse_distance = np.where(
            loop_rows.valid, 1.0 / (loop_rows.distances + _DISTANCE_EPSILON), 0.0)
        weight_sum = np.bincount(
            loop_vertex_indices, weights=inverse_distance, minlength=vertex_count)
        self.valid = weight_sum > 0.0
        safe_sum = np.where(self.valid, weight_sum, 1.0)
        # 逐 loop 归一化收敛权重:顶点采样 = Σ loop 权重 × loop 采样,纯线性可预计算。
        self._loop_weights = inverse_distance / safe_sum[loop_vertex_indices]
        self._loop_vertex_indices = loop_vertex_indices
        self._vertex_count = vertex_count
        self._loop_rows = loop_rows
        # 距离取同顶点各 loop 的加权平均(未命中 loop 权重为 0,inf 先清零防 nan)。
        finite_distances = np.where(loop_rows.valid, loop_rows.distances, 0.0)
        self.distances = np.bincount(
            loop_vertex_indices,
            weights=finite_distances * self._loop_weights,
            minlength=vertex_count)
        self.distances[~self.valid] = np.inf

    def sample(self, data, domain=POINT):
        if domain == CORNER:
            loop_samples = self._loop_rows.sample_corner_data(data)
        else:
            loop_samples = self._loop_rows.sample_vertex_data(data)
        weighted = loop_samples * self._loop_weights[:, None]
        channel_count = weighted.shape[1]
        result = np.zeros((self._vertex_count, channel_count), dtype=np.float64)
        for channel in range(channel_count):
            result[:, channel] = np.bincount(
                self._loop_vertex_indices,
                weights=weighted[:, channel],
                minlength=self._vertex_count)
        return result


class TopologyVertexCorrespondence:
    """拓扑(顶点序号)直通对应:顶点数一致时的逐序号拷贝。"""

    __slots__ = ("valid", "distances", "_bridge")

    def __init__(self, vertex_count, bridge):
        self.valid = np.ones(vertex_count, dtype=bool)
        self.distances = np.zeros(vertex_count, dtype=np.float64)
        self._bridge = bridge

    def sample(self, data, domain=POINT):
        return self._bridge.to_domain(data, domain, POINT).copy()


class DirectCornerCorrespondence:
    """逐角点(loop)三角形插值查询的角点域对应。"""

    __slots__ = ("valid", "distances", "_rows")

    def __init__(self, rows):
        self._rows = rows
        self.valid = rows.valid
        self.distances = rows.distances

    def sample(self, data, domain=CORNER):
        if domain == POINT:
            return self._rows.sample_vertex_data(data)
        return self._rows.sample_corner_data(data)

    def dominant_corners(self):
        return self._rows.dominant_corners()


class TopologyCornerCorrespondence:
    """拓扑直通的角点域对应:loop 数一致时的逐序号拷贝。"""

    __slots__ = ("valid", "distances", "_bridge", "_loop_count")

    def __init__(self, loop_count, bridge):
        self.valid = np.ones(loop_count, dtype=bool)
        self.distances = np.zeros(loop_count, dtype=np.float64)
        self._bridge = bridge
        self._loop_count = loop_count

    def sample(self, data, domain=CORNER):
        return self._bridge.to_domain(data, domain, CORNER).copy()

    def dominant_corners(self):
        return np.arange(self._loop_count, dtype=np.int64)


def match_corners_by_face_values(source_values, source_loop_starts, source_loop_totals,
                                 target_values, target_loop_starts, target_loop_totals,
                                 target_loop_count):
    """按整张面的基准值组合配对源面与目标面,再在面内按值配对角点。

    单个值会撞车(两个顶点共用同一个 UV 坐标,点级无论如何都分不出该去哪一个),
    整张面的值组合撞车则几乎不可能 —— 判据升一级就能解开点级解不开的歧义,
    拓扑一致时逐角点精确。返回 target_loop → source_loop,没配上为 -1。
    """
    source = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    source_rows = [row.tobytes() for row in source]
    target_rows = [row.tobytes() for row in target]

    lookup = {}
    for face in range(source_loop_starts.shape[0]):
        start = int(source_loop_starts[face])
        total = int(source_loop_totals[face])
        key = (total, b"".join(sorted(source_rows[start:start + total])))
        lookup.setdefault(key, (start, total))

    result = np.full(target_loop_count, -1, dtype=np.int64)
    for face in range(target_loop_starts.shape[0]):
        start = int(target_loop_starts[face])
        total = int(target_loop_totals[face])
        key = (total, b"".join(sorted(target_rows[start:start + total])))
        entry = lookup.get(key)
        if entry is None:
            continue
        source_start, source_total = entry
        used = [False] * source_total
        for offset in range(total):
            target_loop = start + offset
            value = target_rows[target_loop]
            for source_offset in range(source_total):
                if used[source_offset]:
                    continue
                if source_rows[source_start + source_offset] == value:
                    used[source_offset] = True
                    result[target_loop] = source_start + source_offset
                    break
    return result


def exact_value_groups(source_values, target_values):
    """把源与目标的基准值按"逐位相同"分组。

    返回 (member_offsets, members, target_group):
      members[member_offsets[g]:member_offsets[g + 1]] = 第 g 组的全部源元素索引,
      target_group[i] = 目标元素 i 落在哪一组(源上没有这个值则 -1)。

    基准匹配的定义就是"值相同即同一处",所以值查找才是第一手依据,几何查询只是
    值对不上时的退路 —— UV 岛重叠处同一坐标被多个源三角形覆盖、命中距离全为 0,
    几何查询挑中哪个纯看 BVH 遍历顺序。同一个值仍可能落在多个源元素上
    (两个岛的顶点撞在同一 UV),那是真歧义,交给调用方按上下文决胜。
    """
    source = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    source_count = source.shape[0]
    target_count = target.shape[0]
    zero = np.zeros(1, dtype=np.int64)
    if source_count == 0 or target_count == 0:
        return (zero, np.empty(0, dtype=np.int64),
                np.full(target_count, -1, dtype=np.int64))
    combined = np.concatenate((source, target), axis=0)
    _unique_rows, inverse_indices = np.unique(combined, axis=0, return_inverse=True)
    inverse_indices = inverse_indices.reshape(-1)
    group_count = int(inverse_indices.max()) + 1
    source_ids = inverse_indices[:source_count]
    members = np.argsort(source_ids, kind='stable')
    counts = np.bincount(source_ids, minlength=group_count)
    member_offsets = np.concatenate((zero, np.cumsum(counts)))
    target_ids = inverse_indices[source_count:]
    return member_offsets, members, np.where(counts[target_ids] > 0, target_ids, -1)


def deduplicate_queries(query_points):
    """把重复的查询点去重(内部顶点的多个 loop 通常共享同一 UV / 同一属性值,
    去重后查询量典型减少 4~6 倍)。返回 (唯一行索引, 逆映射)。

    量化步长按数据幅度自适应:UV(~1)与世界坐标(~1e3)都拿到 ~1e-7 的相对精度,
    所以任意基准通道都能安全共用这一条去重路径。
    """
    values = np.asarray(query_points, dtype=np.float64)
    if values.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    magnitude = float(np.max(np.abs(values)))
    step = 16777216.0 / max(magnitude, 1.0)
    quantized = np.round(values * step).astype(np.int64)
    _unique_rows, first_indices, inverse_indices = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True)
    return first_indices, inverse_indices.reshape(-1)


def snap_positions_to_nearest(positions, reference_positions, valid_mask):
    """可选后处理:把采样位置吸附到最近的源顶点(旧版 Snap 功能的等价保留)。
    只处理有效命中行;返回新数组,不修改输入(修复旧版原地别名副作用)。"""
    reference = np.asarray(reference_positions, dtype=np.float64)
    kd_tree = build_kd_tree(reference)
    snapped = np.array(positions, dtype=np.float64, copy=True)
    find = kd_tree.find
    for index in np.nonzero(valid_mask)[0].tolist():
        location, _reference_index, _distance = find(snapped[index])
        if location is not None:
            snapped[index] = location
    return snapped

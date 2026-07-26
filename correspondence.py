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
                 "_triangle_corner_positions", "_bvh_tree")

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
            角点域传"真实角点"(配合导向偏置查询实现接缝正确 + 边界外推精确)。
        """
        valid = triangle_indices >= 0
        safe_indices = np.where(valid, triangle_indices, 0)
        corners = self._triangle_corner_positions[safe_indices]
        weights = compute_barycentric_weights(weight_points, corners, clamp_inside)
        return CorrespondenceRows(valid, safe_indices, weights, distances, self)


class EdgeNearestQuery:
    """边最近点查询:以退化三角形 (a, b, b) 建 BVH,find_nearest 的命中即线段最近点
    (closest_on_tri 对退化三角形自然退化为线段最近点)。"""

    __slots__ = ("edge_vertex_indices", "_bvh_tree")

    def __init__(self, vertex_positions, edge_vertex_indices):
        self.edge_vertex_indices = edge_vertex_indices  # (E, 2) int64
        triangles = np.column_stack((
            edge_vertex_indices[:, 0],
            edge_vertex_indices[:, 1],
            edge_vertex_indices[:, 1]))
        self._bvh_tree = BVHTree.FromPolygons(
            vertex_positions.tolist(), triangles.tolist(), all_triangles=True)

    def query_nearest(self, query_points, max_distance=None):
        """返回 (edge_indices, hit_positions, distances),未命中行 index=-1。"""
        count = query_points.shape[0]
        edge_indices = np.full(count, -1, dtype=np.int64)
        hit_positions = np.zeros((count, 3), dtype=np.float64)
        distances = np.full(count, np.inf, dtype=np.float64)
        search_radius = float(max_distance) if max_distance is not None else _UNLIMITED_DISTANCE
        find_nearest = self._bvh_tree.find_nearest
        for index, point in enumerate(query_points.tolist()):
            location, _normal, edge_index, distance = find_nearest(point, search_radius)
            if edge_index is not None:
                edge_indices[index] = edge_index
                hit_positions[index] = location
                distances[index] = distance
        return edge_indices, hit_positions, distances


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


class TopologyCornerCorrespondence:
    """拓扑直通的角点域对应:loop 数一致时的逐序号拷贝。"""

    __slots__ = ("valid", "distances", "_bridge")

    def __init__(self, loop_count, bridge):
        self.valid = np.ones(loop_count, dtype=bool)
        self.distances = np.zeros(loop_count, dtype=np.float64)
        self._bridge = bridge

    def sample(self, data, domain=CORNER):
        return self._bridge.to_domain(data, domain, CORNER).copy()


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

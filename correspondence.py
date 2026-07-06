# 表面对应内核:BVH 最近点/射线查询 + 重心插值采样。
# 纯数学层:只依赖 numpy 与 mathutils,不 import bpy。
# 设计要点:
#   1. 所有匹配都是"插值最近表面"——命中三角形 + 重心权重,绝不做最近顶点吸附
#      (吸附只作为可选后处理,见 snap_positions_to_nearest)。
#   2. 同一个内核同时服务 3D 空间(BVH 顶点池 = 网格顶点)与 UV 空间
#      (BVH 顶点池 = 逐角点 UV 升维到 z=0),靠 triangle_vertex_indices /
#      triangle_corner_indices 双索引把命中三角形还原到源网格的顶点域与角点域。
#   3. UV 空间的顶点域采样必须经 CombinedVertexCorrespondence 收敛:
#      同一网格顶点的多个 UV loop 按逆距离权重合并成唯一结果,
#      修复旧版"最后写入者赢"导致的合并顶点/接缝顶点错乱。

import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

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


class CorrespondenceRows:
    """一次查询的逐行命中结果:命中三角形 + 重心权重,可对任意源数据插值采样。"""

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


class SurfaceCorrespondence:
    """源表面的查询结构:一次构建,多次查询/采样。"""

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


class DirectVertexCorrespondence:
    """3D 空间逐顶点直查的顶点域对应:一顶点一命中,天然一致。"""

    __slots__ = ("valid", "distances", "_rows")

    def __init__(self, rows):
        self._rows = rows
        self.valid = rows.valid
        self.distances = rows.distances

    def sample(self, vertex_data):
        return self._rows.sample_vertex_data(vertex_data)


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

    def sample(self, vertex_data):
        loop_samples = self._loop_rows.sample_vertex_data(vertex_data)
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

    __slots__ = ("valid", "distances")

    def __init__(self, vertex_count):
        self.valid = np.ones(vertex_count, dtype=bool)
        self.distances = np.zeros(vertex_count, dtype=np.float64)

    def sample(self, vertex_data):
        return np.asarray(vertex_data, dtype=np.float64).copy()


class DirectCornerCorrespondence:
    """逐角点(loop)查询的角点域对应。"""

    __slots__ = ("valid", "distances", "_rows")

    def __init__(self, rows):
        self._rows = rows
        self.valid = rows.valid
        self.distances = rows.distances

    def sample(self, corner_data):
        return self._rows.sample_corner_data(corner_data)


class TopologyCornerCorrespondence:
    """拓扑直通的角点域对应:loop 数一致时的逐序号拷贝。"""

    __slots__ = ("valid", "distances")

    def __init__(self, loop_count):
        self.valid = np.ones(loop_count, dtype=bool)
        self.distances = np.zeros(loop_count, dtype=np.float64)

    def sample(self, corner_data):
        return np.asarray(corner_data, dtype=np.float64).copy()


def deduplicate_uv_queries(uv_coordinates):
    """按 2^24 量化把重复 UV 查询点去重(内部顶点的多个 loop 通常共享同一 UV,
    去重后查询量典型减少 4~6 倍)。返回 (唯一行索引, 逆映射)。"""
    quantized = np.round(np.asarray(uv_coordinates, dtype=np.float64) * 16777216.0).astype(np.int64)
    # 双通道压成单个 int64 键:|qv| < 2^31 时无碰撞。
    composite_keys = quantized[:, 0] * 4294967296 + quantized[:, 1]
    _unique_keys, first_indices, inverse_indices = np.unique(
        composite_keys, return_index=True, return_inverse=True)
    return first_indices, inverse_indices


def snap_positions_to_nearest(positions, reference_positions, valid_mask):
    """可选后处理:把采样位置吸附到最近的源顶点(旧版 Snap 功能的等价保留)。
    只处理有效命中行;返回新数组,不修改输入(修复旧版原地别名副作用)。"""
    reference = np.asarray(reference_positions, dtype=np.float64)
    kd_tree = KDTree(reference.shape[0])
    insert = kd_tree.insert
    for index, coordinate in enumerate(reference.tolist()):
        insert(coordinate, index)
    kd_tree.balance()
    snapped = np.array(positions, dtype=np.float64, copy=True)
    find = kd_tree.find
    for index in np.nonzero(valid_mask)[0].tolist():
        location, _reference_index, _distance = find(snapped[index])
        if location is not None:
            snapped[index] = location
    return snapped

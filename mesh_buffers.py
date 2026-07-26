# Blender 网格 ⇄ numpy 批量搬运层。
# 铁律:所有批量读写一律走 foreach_get / foreach_set,且 dtype 与 RNA 底层精确一致
# (float→float32、int→int32、bool→bool)以命中 memcpy 快路径,数学计算统一提升 float64。
# 快照(MeshBufferSnapshot)支持 depsgraph 求值结果:源侧所有数组来自同一份网格,
# 修复旧版"BVH 用求值网格、权重用原网格"的索引错位缺陷。

import numpy as np


def matrix_to_numpy(matrix):
    """mathutils.Matrix → (4, 4) float64。"""
    return np.array(matrix, dtype=np.float64)


def transform_points(points, matrix4):
    """仿射变换一批点。points: (N, 3),matrix4: (4, 4)。"""
    return points @ matrix4[:3, :3].T + matrix4[:3, 3]


def transform_directions(vectors, linear3):
    """线性变换一批方向向量(不含平移)。"""
    return vectors @ linear3.T


def normalized_rows(vectors, fallback=None):
    """逐行归一化;零长度行回退到 fallback 对应行(或保持零)。"""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    result = np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 1e-12)
    if fallback is not None and not np.all(valid):
        result[~valid] = fallback[~valid]
    return result


def _read_floats(collection, attribute, item_size):
    count = len(collection)
    buffer = np.empty(count * item_size, dtype=np.float32)
    collection.foreach_get(attribute, buffer)
    if item_size == 1:
        return buffer.astype(np.float64)
    return buffer.astype(np.float64).reshape(count, item_size)


def _read_ints(collection, attribute, item_size):
    count = len(collection)
    buffer = np.empty(count * item_size, dtype=np.int32)
    collection.foreach_get(attribute, buffer)
    if item_size == 1:
        return buffer.astype(np.int64)
    return buffer.astype(np.int64).reshape(count, item_size)


def read_uv_layer(mesh, layer_name):
    """读取 UV 层为 (L, 2) float64;层不存在返回 None。"""
    layer = mesh.uv_layers.get(layer_name)
    if layer is None:
        return None
    return _read_floats(layer.uv, "vector", 2)


def write_uv_layer(mesh, layer_name, uv_coordinates):
    layer = mesh.uv_layers.get(layer_name)
    layer.uv.foreach_set("vector", uv_coordinates.astype(np.float32).ravel())


def ensure_uv_layer(mesh, layer_name, force_new=False):
    """确保目标 UV 层存在。返回实际层名;超出 Blender 8 层上限返回 None。"""
    if not force_new:
        existing = mesh.uv_layers.get(layer_name)
        if existing is not None:
            return existing.name
    if len(mesh.uv_layers) >= 8:
        return None
    created = mesh.uv_layers.new(name=layer_name, do_init=True)
    if created is None:
        return None
    return created.name


def read_color_attribute(mesh, attribute_name):
    """读取颜色属性。返回 (domain, data_type, (N, 4) float64);不存在返回 None。"""
    attribute = mesh.color_attributes.get(attribute_name)
    if attribute is None:
        return None
    values = _read_floats(attribute.data, "color", 4)
    return attribute.domain, attribute.data_type, values


def ensure_color_attribute(mesh, attribute_name, data_type, domain):
    """确保目标颜色属性存在且域/类型与源一致;不一致时重建。返回 (attribute, recreated)。"""
    attribute = mesh.color_attributes.get(attribute_name)
    recreated = False
    if attribute is not None and (attribute.domain != domain or attribute.data_type != data_type):
        mesh.color_attributes.remove(attribute)
        attribute = None
        recreated = True
    if attribute is None:
        attribute = mesh.color_attributes.new(name=attribute_name, type=data_type, domain=domain)
    return attribute, recreated


def write_color_attribute(attribute, colors):
    attribute.data.foreach_set("color", colors.astype(np.float32).ravel())


# ---------- 通用属性(mesh.attributes 全域) ----------

# data_type → (RNA 值属性名, 分量数, 数值种类)。未列出的类型(STRING/FLOAT4X4)不支持。
_ATTRIBUTE_LAYOUT = {
    'FLOAT': ("value", 1, 'FLOAT'),
    'FLOAT_VECTOR': ("vector", 3, 'FLOAT'),
    'FLOAT2': ("vector", 2, 'FLOAT'),
    'FLOAT_COLOR': ("color", 4, 'FLOAT'),
    'BYTE_COLOR': ("color", 4, 'FLOAT'),
    'QUATERNION': ("value", 4, 'FLOAT'),
    'INT': ("value", 1, 'INT'),
    'INT8': ("value", 1, 'INT'),
    'INT32_2D': ("value", 2, 'INT'),
    'BOOLEAN': ("value", 1, 'BOOL'),
}

# 可作为转换写入目标的属性类型(供 UI 枚举)。
ATTRIBUTE_TYPE_ITEMS = [
    ('FLOAT', "Float", "Single float value", 0),
    ('FLOAT_VECTOR', "Vector", "3D float vector", 1),
    ('FLOAT2', "2D Vector", "2D float vector", 2),
    ('FLOAT_COLOR', "Color", "RGBA float color", 3),
    ('BYTE_COLOR', "Byte Color", "RGBA 8-bit color", 4),
    ('INT', "Integer", "Single integer value", 5),
    ('BOOLEAN', "Boolean", "True/false value", 6),
]


def attribute_component_count(data_type):
    layout = _ATTRIBUTE_LAYOUT.get(data_type)
    return layout[1] if layout is not None else 0


def read_generic_attribute(mesh, attribute_name):
    """读取任意网格属性。返回 (domain, data_type, (N, C) float64);不支持/不存在返回 None。"""
    attribute = mesh.attributes.get(attribute_name)
    if attribute is None:
        return None
    layout = _ATTRIBUTE_LAYOUT.get(attribute.data_type)
    if layout is None:
        return None
    value_attribute, item_size, kind = layout
    count = len(attribute.data)
    if kind == 'FLOAT':
        buffer = np.empty(count * item_size, dtype=np.float32)
    elif kind == 'INT':
        buffer = np.empty(count * item_size, dtype=np.int32)
    else:
        buffer = np.empty(count * item_size, dtype=bool)
    try:
        attribute.data.foreach_get(value_attribute, buffer)
    except (RuntimeError, TypeError, AttributeError):
        return None
    values = buffer.astype(np.float64).reshape(count, item_size)
    return attribute.domain, attribute.data_type, values


def ensure_generic_attribute(mesh, attribute_name, data_type, domain):
    """确保属性存在且域/类型一致;不一致时重建。返回 (attribute, recreated);无法建立返回 (None, False)。"""
    attribute = mesh.attributes.get(attribute_name)
    recreated = False
    if attribute is not None and (attribute.domain != domain
                                  or attribute.data_type != data_type):
        try:
            mesh.attributes.remove(attribute)
        except RuntimeError:
            return None, False  # 内建属性(position 等)不可重建
        attribute = None
        recreated = True
    if attribute is None:
        try:
            attribute = mesh.attributes.new(
                name=attribute_name, type=data_type, domain=domain)
        except (RuntimeError, TypeError):
            return None, recreated
    return attribute, recreated


def write_generic_attribute(attribute, values):
    """写回任意网格属性,按其数值种类做精确 dtype 转换。"""
    layout = _ATTRIBUTE_LAYOUT.get(attribute.data_type)
    if layout is None:
        return False
    value_attribute, _item_size, kind = layout
    if kind == 'FLOAT':
        buffer = values.astype(np.float32).ravel()
    elif kind == 'INT':
        buffer = np.rint(values).astype(np.int32).ravel()
    else:
        buffer = (values > 0.5).ravel()
    try:
        attribute.data.foreach_set(value_attribute, buffer)
    except (RuntimeError, TypeError):
        return False
    return True


def write_vertex_positions(mesh, positions):
    mesh.vertices.foreach_set("co", positions.astype(np.float32).ravel())
    mesh.update()


def apply_vertex_positions(target_object, positions):
    """把顶点移动到新坐标,并保住形态键。

    目标带形态键时:Basis 与全部键整体平移同样的位移,各键的相对形变原样保留
    (只写网格顶点在有键时视口不生效,是旧版的隐性缺陷)。
    """
    mesh = target_object.data
    if mesh.shape_keys is not None:
        key_blocks = mesh.shape_keys.key_blocks
        vertex_count = len(mesh.vertices)
        basis_positions = read_shape_key_positions(key_blocks[0], vertex_count)
        shift = positions - basis_positions
        for key_block in key_blocks:
            key_positions = read_shape_key_positions(key_block, vertex_count)
            write_shape_key_positions(key_block, key_positions + shift)
    write_vertex_positions(mesh, positions)


def ensure_shape_key(target_object, shape_key_name):
    """确保目标对象存在 Basis 与指定形态键。返回 (key_block, created)。
    新建键的初始坐标 = 当前基础网格(from_mix=False)。"""
    mesh = target_object.data
    if mesh.shape_keys is None:
        target_object.shape_key_add(name="Basis", from_mix=False)
    key_block = mesh.shape_keys.key_blocks.get(shape_key_name)
    if key_block is not None:
        return key_block, False
    key_block = target_object.shape_key_add(name=shape_key_name, from_mix=False)
    return key_block, True


def read_shape_key_positions(key_block, vertex_count):
    buffer = np.empty(vertex_count * 3, dtype=np.float32)
    key_block.data.foreach_get("co", buffer)
    return buffer.astype(np.float64).reshape(vertex_count, 3)


def write_shape_key_positions(key_block, positions):
    key_block.data.foreach_set("co", positions.astype(np.float32).ravel())


def write_corner_normals(mesh, normals):
    """写入自定义拆分法线(角点域)。"""
    mesh.normals_split_custom_set(normals.astype(np.float32).tolist())
    mesh.update()


def write_vertex_group_weights(target_object, group_name, weights):
    """整组重写顶点组权重。

    原地清空再写入,保住组顺序与 lock_weight 标记(旧版删组重建会破坏两者)。
    权重按 16 位量化(最大误差 ≈ 7.6e-6)后对相同值批量 add,把 O(V) 次
    Python→C 调用压缩到 O(不同权重数) 次。
    """
    vertex_count = weights.shape[0]
    all_indices = np.arange(vertex_count, dtype=np.int64)
    vertex_group = target_object.vertex_groups.get(group_name)
    if vertex_group is None:
        vertex_group = target_object.vertex_groups.new(name=group_name)
    else:
        vertex_group.remove(all_indices.tolist())
    quantized = np.round(np.clip(weights, 0.0, 1.0) * 65535.0).astype(np.int64)
    nonzero_mask = quantized > 0
    if not np.any(nonzero_mask):
        return
    values = quantized[nonzero_mask]
    indices = all_indices[nonzero_mask]
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    sorted_indices = indices[order]
    boundaries = np.nonzero(np.diff(sorted_values))[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_values.shape[0]]))
    add = vertex_group.add
    for start, end in zip(starts.tolist(), ends.tolist()):
        add(sorted_indices[start:end].tolist(), sorted_values[start] / 65535.0, 'REPLACE')


class MeshBufferSnapshot:
    """一个对象(可选 depsgraph 求值结果)的网格数据快照,所有数组懒加载并缓存。

    use_evaluated=True 时:BVH 几何、UV、颜色、法线、权重全部来自同一份求值网格,
    保证跨数组索引一致(修复旧版求值几何 × 原始权重的错位)。
    注意:求值网格指针在下一次 depsgraph 更新后失效,持有期间不得触发源对象重求值;
    需要重求值的流程(形态键隔离快照)必须安排在所有懒加载完成之后(见 conform_session)。
    """

    def __init__(self, source_object, use_evaluated=False, depsgraph=None):
        self.object = source_object
        self._evaluated_object = None
        if use_evaluated and depsgraph is not None:
            self._evaluated_object = source_object.evaluated_get(depsgraph)
            self.mesh = self._evaluated_object.to_mesh(
                preserve_all_data_layers=True, depsgraph=depsgraph)
        else:
            self.mesh = source_object.data
        self._cache = {}

    def free(self):
        """释放求值网格。求值对象可能已随 depsgraph 更新失效,静默容错。"""
        if self._evaluated_object is not None:
            try:
                self._evaluated_object.to_mesh_clear()
            except (ReferenceError, RuntimeError):
                pass
            self._evaluated_object = None
        self.mesh = None
        self._cache.clear()

    def _cached(self, key, builder):
        value = self._cache.get(key)
        if value is None:
            value = builder()
            self._cache[key] = value
        return value

    # ---------- 拓扑 ----------

    @property
    def vertex_count(self):
        return len(self.mesh.vertices)

    @property
    def loop_count(self):
        return len(self.mesh.loops)

    @property
    def vertex_positions(self):
        return self._cached(
            "vertex_positions", lambda: _read_floats(self.mesh.vertices, "co", 3))

    @property
    def vertex_normals(self):
        return self._cached(
            "vertex_normals", lambda: _read_floats(self.mesh.vertex_normals, "vector", 3))

    @property
    def vertex_selection(self):
        def build():
            buffer = np.empty(self.vertex_count, dtype=bool)
            self.mesh.vertices.foreach_get("select", buffer)
            return buffer
        return self._cached("vertex_selection", build)

    @property
    def loop_vertex_indices(self):
        return self._cached(
            "loop_vertex_indices", lambda: _read_ints(self.mesh.loops, "vertex_index", 1))

    def _build_loop_triangles(self):
        mesh = self.mesh
        mesh.calc_loop_triangles()
        triangle_count = len(mesh.loop_triangles)
        vertex_buffer = np.empty(triangle_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", vertex_buffer)
        loop_buffer = np.empty(triangle_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", loop_buffer)
        polygon_buffer = np.empty(triangle_count, dtype=np.int32)
        mesh.loop_triangles.foreach_get("polygon_index", polygon_buffer)
        return (vertex_buffer.astype(np.int64).reshape(triangle_count, 3),
                loop_buffer.astype(np.int64).reshape(triangle_count, 3),
                polygon_buffer.astype(np.int64))

    @property
    def triangle_vertex_indices(self):
        return self._cached("loop_triangles", self._build_loop_triangles)[0]

    @property
    def triangle_loop_indices(self):
        return self._cached("loop_triangles", self._build_loop_triangles)[1]

    @property
    def triangle_polygon_indices(self):
        """每个 loop 三角形所属的多边形索引,(T,)。"""
        return self._cached("loop_triangles", self._build_loop_triangles)[2]

    @property
    def edge_vertex_indices(self):
        """(E, 2) 边端点索引。"""
        return self._cached(
            "edge_vertex_indices", lambda: _read_ints(self.mesh.edges, "vertices", 2))

    @property
    def polygon_normals(self):
        return self._cached(
            "polygon_normals", lambda: _read_floats(self.mesh.polygon_normals, "vector", 3))

    @property
    def polygon_loop_starts(self):
        return self._cached(
            "polygon_loop_starts", lambda: _read_ints(self.mesh.polygons, "loop_start", 1))

    @property
    def polygon_loop_totals(self):
        return self._cached(
            "polygon_loop_totals", lambda: _read_ints(self.mesh.polygons, "loop_total", 1))

    @property
    def loop_polygon_indices(self):
        """每个 loop 所属的多边形索引,(L,)。"""
        def build():
            loop_starts = self.polygon_loop_starts
            loop_totals = self.polygon_loop_totals
            # 各面 loop 区间按 loop_start 升序恰好铺满 [0, L),按序 repeat 即得 loop→face。
            order = np.argsort(loop_starts, kind='stable')
            return np.repeat(order, loop_totals[order])
        return self._cached("loop_polygon_indices", build)

    @property
    def corner_face_centers(self):
        """每个 loop 所属面的中心坐标,(L, 3)。用于角点导向偏置采样。"""
        def build():
            centers = _read_floats(self.mesh.polygons, "center", 3)
            return centers[self.loop_polygon_indices]
        return self._cached("corner_face_centers", build)

    @property
    def vertex_loop_csr(self):
        """顶点 → 其全部 loop 的 CSR 反查表:(按顶点分组排序的 loop 索引, 段偏移 (V+1,))。"""
        def build():
            loop_vertex = self.loop_vertex_indices
            order = np.argsort(loop_vertex, kind='stable')
            counts = np.bincount(loop_vertex, minlength=self.vertex_count)
            offsets = np.concatenate(([0], np.cumsum(counts)))
            return order, offsets
        return self._cached("vertex_loop_csr", build)

    @property
    def corner_normals(self):
        return self._cached(
            "corner_normals", lambda: _read_floats(self.mesh.corner_normals, "vector", 3))

    # ---------- UV ----------

    @property
    def uv_layer_names(self):
        return [layer.name for layer in self.mesh.uv_layers]

    @property
    def active_uv_layer_name(self):
        active = self.mesh.uv_layers.active
        return active.name if active is not None else None

    def read_uv_layer(self, layer_name):
        return self._cached(
            ("uv_layer", layer_name), lambda: read_uv_layer(self.mesh, layer_name))

    # ---------- 颜色属性 ----------

    @property
    def color_attribute_names(self):
        return [attribute.name for attribute in self.mesh.color_attributes]

    @property
    def active_color_attribute_name(self):
        active = self.mesh.color_attributes.active_color
        return active.name if active is not None else None

    def read_color_attribute(self, attribute_name):
        return self._cached(
            ("color_attribute", attribute_name),
            lambda: read_color_attribute(self.mesh, attribute_name))

    # ---------- 通用属性 ----------

    @property
    def attribute_names(self):
        """可读的用户属性名(过滤掉 '.' 开头的内部属性)。"""
        return [attribute.name for attribute in self.mesh.attributes
                if not attribute.name.startswith(".")]

    @property
    def active_attribute_name(self):
        active = self.mesh.attributes.active
        return active.name if active is not None else None

    def read_generic_attribute(self, attribute_name):
        return self._cached(
            ("attribute", attribute_name),
            lambda: read_generic_attribute(self.mesh, attribute_name))

    # ---------- 形态键 ----------

    @property
    def shape_key_names(self):
        """形态键始终读自原始对象数据(求值网格不携带形态键数据块)。"""
        shape_keys = self.object.data.shape_keys
        if shape_keys is None:
            return []
        return [key_block.name for key_block in shape_keys.key_blocks]

    @property
    def active_shape_key_name(self):
        shape_keys = self.object.data.shape_keys
        if shape_keys is None:
            return None
        index = self.object.active_shape_key_index
        key_blocks = shape_keys.key_blocks
        if 0 <= index < len(key_blocks):
            return key_blocks[index].name
        return None

    def read_shape_key(self, shape_key_name):
        """读取指定形态键坐标 (V, 3);不存在或顶点数与快照不符返回 None。"""
        def build():
            shape_keys = self.object.data.shape_keys
            if shape_keys is None:
                return None
            key_block = shape_keys.key_blocks.get(shape_key_name)
            if key_block is None:
                return None
            count = len(key_block.data)
            if count != self.vertex_count:
                return None
            return read_shape_key_positions(key_block, count)
        return self._cached(("shape_key", shape_key_name), build)

    # ---------- 顶点组 ----------

    @property
    def vertex_group_names(self):
        return [group.name for group in self.object.vertex_groups]

    @property
    def vertex_group_locks(self):
        return [group.lock_weight for group in self.object.vertex_groups]

    @property
    def vertex_group_weight_matrix(self):
        """(V, G) 权重矩阵,一趟遍历读出全部组(求值网格的 deform 层索引与对象组序一致)。"""
        def build():
            group_count = len(self.object.vertex_groups)
            matrix = np.zeros((self.vertex_count, group_count), dtype=np.float64)
            if group_count == 0:
                return matrix
            for vertex in self.mesh.vertices:
                vertex_index = vertex.index
                for element in vertex.groups:
                    group_index = element.group
                    if 0 <= group_index < group_count:
                        matrix[vertex_index, group_index] = element.weight
            return matrix
        return self._cached("vertex_group_weight_matrix", build)

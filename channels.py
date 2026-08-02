# 通道层:把"网格上任意一段数据"统一成 (N, C) float64 + 域(POINT/CORNER),
# 于是"任意通道 → 任意通道"只剩四步:读源 → 对应关系采样 → 数值/分量重映射 → 写目标。
# 覆盖通道:顶点位置 / 形态键 / UV / 自定义法线 / 颜色属性 / 顶点组权重 / 通用属性。
# 位置类通道(位置/形态键)只交出局部坐标,空间变换一律由会话层统一处理,
# 本层不碰任何矩阵,保证"数值语义"与"坐标空间"两件事互不污染。

import numpy as np

from .correspondence import POINT, CORNER
from .mesh_buffers import (
    add_numbered_uv_layer,
    attribute_component_count,
    ensure_color_attribute,
    ensure_generic_attribute,
    ensure_shape_key,
    ensure_uv_layer,
    normalized_rows,
    read_color_attribute,
    read_generic_attribute,
    read_shape_key_positions,
    read_uv_layer,
    apply_vertex_positions,
    write_color_attribute,
    write_corner_normals,
    write_generic_attribute,
    write_shape_key_positions,
    write_uv_layer,
    write_vertex_group_weights,
)


class ChannelError(RuntimeError):
    """通道读写前提不满足(层不存在、类型不支持等),由会话层转成 ConformError。"""


# ==================== 通道定义 ====================

POSITION = 'POSITION'
SHAPE_KEY = 'SHAPE_KEY'
UV = 'UV'
NORMAL = 'NORMAL'
COLOR = 'COLOR'
WEIGHT = 'WEIGHT'
ATTRIBUTE = 'ATTRIBUTE'

# (标识符, 名称, 说明, 图标, 序号) —— 源端可读的全部通道。
SOURCE_CHANNEL_ITEMS = [
    (POSITION, "Vertex Position",
     "Vertex coordinates", 'MESH_DATA', 0),
    (SHAPE_KEY, "Shape Key",
     "Coordinates of one shape key", 'SHAPEKEY_DATA', 1),
    (UV, "UV Map",
     "UV coordinates of one UV layer", 'UV', 2),
    (NORMAL, "Normal",
     "Final split normals per face corner", 'NORMALS_VERTEX_FACE', 3),
    (COLOR, "Color Attribute",
     "One color attribute", 'COLOR', 4),
    (WEIGHT, "Vertex Group",
     "Weights of one vertex group", 'GROUP_VERTEX', 5),
    (ATTRIBUTE, "Attribute",
     "Any point or face-corner attribute", 'MESH_DATA', 6),
]

# 目标端可写的全部通道(顺序把"形态键"排在位置之前:顶点结果默认落形态键)。
TARGET_CHANNEL_ITEMS = [
    (SHAPE_KEY, "Shape Key",
     "Write into a shape key — the mesh itself stays untouched", 'SHAPEKEY_DATA', 0),
    (POSITION, "Vertex Position",
     "Move the mesh vertices themselves", 'MESH_DATA', 1),
    (UV, "UV Map",
     "Write into a UV layer", 'UV', 2),
    (NORMAL, "Normal",
     "Write as custom split normals", 'NORMALS_VERTEX_FACE', 3),
    (COLOR, "Color Attribute",
     "Write into a color attribute", 'COLOR', 4),
    (WEIGHT, "Vertex Group",
     "Write into a vertex group", 'GROUP_VERTEX', 5),
    (ATTRIBUTE, "Attribute",
     "Write into a point or face-corner attribute", 'MESH_DATA', 6),
]

_CHANNEL_LABELS = {identifier: label
                   for identifier, label, _description, _icon, _number
                   in SOURCE_CHANNEL_ITEMS}

# 域固定的通道(COLOR / ATTRIBUTE 的域随各自的属性走)。
_FIXED_DOMAIN = {
    POSITION: POINT,
    SHAPE_KEY: POINT,
    WEIGHT: POINT,
    UV: CORNER,
    NORMAL: CORNER,
}

# 分量数固定的通道(ATTRIBUTE 随其数据类型走)。
_FIXED_COMPONENTS = {
    POSITION: 3,
    SHAPE_KEY: 3,
    WEIGHT: 1,
    UV: 2,
    NORMAL: 3,
    COLOR: 4,
}

# 需要指定层/组/键名的通道(空名 = 用活动项)。
NAMED_CHANNELS = {SHAPE_KEY, UV, COLOR, WEIGHT, ATTRIBUTE}

# 位置类通道:数值本身是坐标,跨对象要经仿射变换。
POSITIONAL_CHANNELS = {POSITION, SHAPE_KEY}
# 方向类通道:数值是方向向量,跨对象只经线性部分的逆转置并重新归一化。
DIRECTIONAL_CHANNELS = {NORMAL}


def channel_label(kind):
    return _CHANNEL_LABELS.get(kind, kind)


def channel_domain(kind):
    """通道的固定域;COLOR/ATTRIBUTE 返回 None(随属性而定)。"""
    return _FIXED_DOMAIN.get(kind)


def channel_components(kind):
    """通道的固定分量数;ATTRIBUTE 返回 None(随数据类型而定)。"""
    return _FIXED_COMPONENTS.get(kind)


# ==================== 分量重映射 ====================

# 目标每个分量的取值来源。
COMPONENT_SOURCE_ITEMS = [
    ('X', "X / R / U", "1st source component", 0),
    ('Y', "Y / G / V", "2nd source component", 1),
    ('Z', "Z / B", "3rd source component", 2),
    ('W', "W / A", "4th source component", 3),
    ('LENGTH', "Length", "Vector length of the source components", 4),
    ('AVERAGE', "Average", "Average of the source components", 5),
    ('ZERO', "0.0", "Constant zero", 6),
    ('ONE', "1.0", "Constant one", 7),
]

_COMPONENT_INDEX = {'X': 0, 'Y': 1, 'Z': 2, 'W': 3}


def _component_column(values, pick, row_count):
    if pick == 'ZERO':
        return np.zeros(row_count, dtype=np.float64)
    if pick == 'ONE':
        return np.ones(row_count, dtype=np.float64)
    if pick == 'LENGTH':
        return np.linalg.norm(values, axis=1)
    if pick == 'AVERAGE':
        return values.mean(axis=1)
    index = _COMPONENT_INDEX[pick]
    if index < values.shape[1]:
        return values[:, index]
    return np.zeros(row_count, dtype=np.float64)


def adapt_components(values, target_components, picks=None):
    """把 (N, C) 采样结果整形成 (N, target_components)。

    picks 非空 = 逐分量显式取值(COMPONENT_SOURCE_ITEMS 标识符列表)。
    picks 为空 = 自动规则(第 4 分量一律按 Alpha 对待,不参与平均、缺省补 1):
      分量数相同        → 原样;
      目标只要 1 个分量 → 取前三个分量的平均(颜色→权重 = 灰度,忽略 Alpha);
      源只有 1 个分量   → 广播(权重→颜色 = 灰阶,Alpha 补 1);
      目标分量更少      → 取前若干个(位置→UV = XY);
      目标分量更多      → 补 0,4 分量目标的末位补 1(UV→位置 = XY0)。
    """
    values = np.asarray(values, dtype=np.float64)
    row_count = values.shape[0]
    source_components = values.shape[1]

    if picks:
        columns = [_component_column(values, pick, row_count)
                   for pick in picks[:target_components]]
        return np.stack(columns, axis=1)

    if source_components == target_components:
        return values
    if target_components == 1:
        return values[:, :min(3, source_components)].mean(axis=1, keepdims=True)
    result = np.zeros((row_count, target_components), dtype=np.float64)
    if source_components == 1:
        result[:, :min(3, target_components)] = values
    elif target_components < source_components:
        return values[:, :target_components]
    else:
        result[:, :source_components] = values
    if target_components == 4:
        result[:, 3] = 1.0
    return result


# ==================== 数值重映射 ====================

VALUE_REMAP_ITEMS = [
    ('NONE', "None", "Use the sampled values as they are", 0),
    ('NORMALIZE', "Normalize 0..1",
     "Fit the sampled value range into 0..1 (bake positions into a color)", 1),
    ('SIGNED_TO_UNIT', "-1..1 to 0..1",
     "Half-and-offset, the usual normal-map encoding", 2),
    ('UNIT_TO_SIGNED', "0..1 to -1..1",
     "Decode a normal-map style value back to a signed vector", 3),
    ('CUSTOM', "Scale + Offset", "Multiply then add the values below", 4),
]


def remap_values(values, mode, scale=1.0, offset=0.0):
    """在分量重映射之前作用于采样值,常量分量因此不受影响。"""
    if mode == 'NONE':
        return values
    if mode == 'SIGNED_TO_UNIT':
        return values * 0.5 + 0.5
    if mode == 'UNIT_TO_SIGNED':
        return values * 2.0 - 1.0
    if mode == 'CUSTOM':
        return values * scale + offset
    minimum = float(np.min(values)) if values.size else 0.0
    maximum = float(np.max(values)) if values.size else 0.0
    span = maximum - minimum
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / span


# ==================== 源端读取 ====================

class ChannelData:
    """一段读出来的源通道:数值 + 域 + 出处。"""

    __slots__ = ("values", "domain", "name", "data_type", "positional", "directional")

    def __init__(self, values, domain, name, data_type=None, positional=False,
                 directional=False):
        self.values = values
        self.domain = domain
        self.name = name
        self.data_type = data_type
        self.positional = positional
        self.directional = directional

    @property
    def components(self):
        return self.values.shape[1]


def lift_to_match_space(values):
    """任意分量数的通道值 → (N, 3) 匹配空间点。

    不足 3 个分量补 0(UV 升到 z=0 平面),超过 3 个只取前 3 个(颜色取 RGB,Alpha 不参与匹配)。
    源与目标用的是同一个通道,所以量纲天然一致,不需要任何归一化。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] == 3:
        return values
    lifted = np.zeros((values.shape[0], 3), dtype=np.float64)
    kept = min(3, values.shape[1])
    lifted[:, :kept] = values[:, :kept]
    return lifted


def channel_names(snapshot, kind):
    """该通道在这个网格上的全部名字(供"按名字成批配对"与面板列举)。"""
    if kind == UV:
        return list(snapshot.uv_layer_names)
    if kind == COLOR:
        return list(snapshot.color_attribute_names)
    if kind == WEIGHT:
        return list(snapshot.vertex_group_names)
    if kind == ATTRIBUTE:
        return list(snapshot.attribute_names)
    if kind == SHAPE_KEY:
        # Basis 是静止态,成批配对时不算一条形变数据。
        return list(snapshot.shape_key_names[1:])
    return []


def resolve_match_name(snapshot, kind, requested_name, preferred_name=""):
    """匹配基准在目标侧的取名:显式指定 > 与源同名 > 该网格的活动项。"""
    if requested_name:
        return requested_name
    if preferred_name and preferred_name in channel_names(snapshot, kind):
        return preferred_name
    return resolve_source_name(snapshot, kind, "")


def resolve_source_name(snapshot, kind, requested_name):
    """把空名解析成活动层/组/键;找不到返回 None。"""
    if kind not in NAMED_CHANNELS:
        return ""
    if requested_name:
        return requested_name
    if kind == UV:
        return snapshot.active_uv_layer_name
    if kind == COLOR:
        return snapshot.active_color_attribute_name
    if kind == ATTRIBUTE:
        return snapshot.active_attribute_name
    if kind == SHAPE_KEY:
        return snapshot.active_shape_key_name
    if kind == WEIGHT:
        names = snapshot.vertex_group_names
        return names[0] if names else None
    return None


def read_source_channel(snapshot, kind, requested_name, side="Source"):
    """读取一个网格上的通道 → ChannelData(局部空间,未做任何变换)。

    side 只影响报错措辞(源侧读数据 / 目标侧读匹配基准),不影响任何行为。
    """
    name = resolve_source_name(snapshot, kind, requested_name)
    if kind in NAMED_CHANNELS and not name:
        raise ChannelError(
            f"{side} has no {channel_label(kind).lower()} to read from")

    if kind == POSITION:
        return ChannelData(snapshot.vertex_positions, POINT, "Position",
                           positional=True)

    if kind == SHAPE_KEY:
        positions = snapshot.read_shape_key(name)
        if positions is None:
            if name in snapshot.shape_key_names:
                raise ChannelError(
                    f"{side} shape key '{name}' does not match the evaluated vertex "
                    f"count — turn Use Modified Source off to read it")
            raise ChannelError(f"{side} shape key '{name}' not found")
        return ChannelData(positions, POINT, name, positional=True)

    if kind == UV:
        uv_coordinates = snapshot.read_uv_layer(name)
        if uv_coordinates is None:
            raise ChannelError(f"{side} UV layer '{name}' not found")
        return ChannelData(uv_coordinates, CORNER, name)

    if kind == NORMAL:
        return ChannelData(snapshot.corner_normals, CORNER, "Normal",
                           directional=True)

    if kind == COLOR:
        payload = snapshot.read_color_attribute(name)
        if payload is None:
            raise ChannelError(f"{side} color attribute '{name}' not found")
        domain, data_type, values = payload
        if domain not in (POINT, CORNER):
            raise ChannelError(
                f"Color attribute '{name}' uses unsupported domain '{domain}'")
        return ChannelData(values, domain, name, data_type)

    if kind == WEIGHT:
        group_names = snapshot.vertex_group_names
        if name not in group_names:
            raise ChannelError(f"{side} vertex group '{name}' not found")
        column = snapshot.vertex_group_weight_matrix[:, group_names.index(name)]
        return ChannelData(column[:, None], POINT, name)

    payload = snapshot.read_generic_attribute(name)
    if payload is None:
        raise ChannelError(
            f"{side} attribute '{name}' not found or uses an unsupported type")
    domain, data_type, values = payload
    if domain not in (POINT, CORNER):
        raise ChannelError(
            f"Attribute '{name}' lives on the '{domain}' domain, which cannot be "
            f"converted (point and face corner only)")
    return ChannelData(values, domain, name, data_type)


# ==================== 目标端写入 ====================

def _auto_attribute_type(components):
    if components <= 1:
        return 'FLOAT'
    if components == 2:
        return 'FLOAT2'
    if components == 3:
        return 'FLOAT_VECTOR'
    return 'FLOAT_COLOR'


class TargetChannel:
    """已就位的目标通道:知道自己的域/分量数,能回读现值、能写回混合结果。

    构造即完成"确保存在"(建层/建组/建键/建属性),因此现值总是可读,
    影响权重混合(mix/遮罩/选择/衰减)对新建通道也成立。
    """

    __slots__ = ("kind", "name", "domain", "components", "object", "mesh",
                 "created", "recreated", "_key_block", "_attribute")

    def __init__(self, target_object, kind, name, source_domain, source_components,
                 domain_mode='AUTO', attribute_type='AUTO'):
        self.kind = kind
        self.object = target_object
        self.mesh = target_object.data
        self.name = name
        self.created = False
        self.recreated = False
        self._key_block = None
        self._attribute = None

        fixed_domain = channel_domain(kind)
        if fixed_domain is not None:
            self.domain = fixed_domain
        elif domain_mode in (POINT, CORNER):
            self.domain = domain_mode
        else:
            self.domain = source_domain
        self.components = channel_components(kind) or 0

        if kind in (POSITION, NORMAL):
            self.name = channel_label(kind)
        elif not self.name:
            raise ChannelError(f"Give the target {channel_label(kind).lower()} a name")

        if self.domain == CORNER and len(self.mesh.loops) == 0:
            raise ChannelError("Target mesh has no face corners")

        if kind == SHAPE_KEY:
            self._key_block, self.created = ensure_shape_key(target_object, self.name)
        elif kind == UV:
            # 与形态键同样的规矩:每次转换新建一层,绝不覆盖上一次的结果。
            self.created = True
            actual_name = add_numbered_uv_layer(self.mesh, self.name)
            if actual_name is None:
                raise ChannelError("Target already has the maximum of 8 UV layers")
            self.name = actual_name
        elif kind == COLOR:
            existing = self.mesh.color_attributes.get(self.name)
            data_type = existing.data_type if existing is not None else 'FLOAT_COLOR'
            if existing is not None:
                self.domain = existing.domain
            attribute, self.recreated = ensure_color_attribute(
                self.mesh, self.name, data_type, self.domain)
            self.created = existing is None
            self._attribute = attribute
            self.name = attribute.name
        elif kind == WEIGHT:
            self.created = self.object.vertex_groups.get(self.name) is None
        elif kind == ATTRIBUTE:
            existing = self.mesh.attributes.get(self.name)
            if existing is not None:
                self.domain = existing.domain
                data_type = existing.data_type
            elif attribute_type != 'AUTO':
                data_type = attribute_type
            else:
                data_type = _auto_attribute_type(source_components)
            attribute, self.recreated = ensure_generic_attribute(
                self.mesh, self.name, data_type, self.domain)
            if attribute is None:
                raise ChannelError(
                    f"Target attribute '{self.name}' cannot be created or replaced")
            self.created = existing is None
            self._attribute = attribute
            self.name = attribute.name
            self.components = attribute_component_count(attribute.data_type)
            if self.components == 0:
                raise ChannelError(
                    f"Target attribute '{self.name}' uses an unsupported type")

    def read_existing(self, snapshot=None):
        """回读目标现值 (N, C) —— 混合的基线。

        顶点位置与顶点组走目标快照的缓存(权重矩阵是唯一逐顶点 Python 遍历,不重复付费),
        其余通道直接 foreach_get 现读,保证刚建好的层也能拿到正确初值。
        """
        if self.kind == POSITION:
            if snapshot is not None:
                return snapshot.vertex_positions
            count = len(self.mesh.vertices)
            buffer = np.empty(count * 3, dtype=np.float32)
            self.mesh.vertices.foreach_get("co", buffer)
            return buffer.astype(np.float64).reshape(count, 3)
        if self.kind == SHAPE_KEY:
            return read_shape_key_positions(self._key_block, len(self.mesh.vertices))
        if self.kind == UV:
            return read_uv_layer(self.mesh, self.name)
        if self.kind == NORMAL:
            corner_normals = self.mesh.corner_normals
            count = len(corner_normals)
            buffer = np.empty(count * 3, dtype=np.float32)
            corner_normals.foreach_get("vector", buffer)
            return buffer.astype(np.float64).reshape(count, 3)
        if self.kind == COLOR:
            return read_color_attribute(self.mesh, self.name)[2]
        if self.kind == WEIGHT:
            vertex_count = len(self.mesh.vertices)
            group = self.object.vertex_groups.get(self.name)
            if group is None:
                return np.zeros((vertex_count, 1), dtype=np.float64)
            if snapshot is not None:
                group_names = snapshot.vertex_group_names
                if self.name in group_names:
                    column = snapshot.vertex_group_weight_matrix[
                        :, group_names.index(self.name)]
                    return column[:, None]
            weights = np.zeros((vertex_count, 1), dtype=np.float64)
            group_index = group.index
            for vertex in self.mesh.vertices:
                for element in vertex.groups:
                    if element.group == group_index:
                        weights[vertex.index, 0] = element.weight
                        break
            return weights
        return read_generic_attribute(self.mesh, self.name)[2]

    def write(self, values):
        """写回最终值。位置类通道由调用方保证已在目标局部空间。"""
        if self.kind == POSITION:
            apply_vertex_positions(self.object, values)
        elif self.kind == SHAPE_KEY:
            write_shape_key_positions(self._key_block, values)
            if self.created:
                self._key_block.value = 1.0
        elif self.kind == UV:
            write_uv_layer(self.mesh, self.name, values)
        elif self.kind == NORMAL:
            write_corner_normals(self.mesh, normalized_rows(values))
        elif self.kind == COLOR:
            if self._attribute.data_type == 'BYTE_COLOR':
                values = np.clip(values, 0.0, 1.0)
            write_color_attribute(self._attribute, values)
            if self.mesh.color_attributes.active_color_index < 0:
                self.mesh.color_attributes.active_color_index = 0
        elif self.kind == WEIGHT:
            write_vertex_group_weights(self.object, self.name, values[:, 0])
        else:
            if not write_generic_attribute(self._attribute, values):
                raise ChannelError(
                    f"Target attribute '{self.name}' could not be written")
        self.mesh.update()

    def focus(self):
        """把刚写好的通道设为活动项,结果一眼可见。"""
        if self.kind == SHAPE_KEY:
            key_blocks = self.mesh.shape_keys.key_blocks
            self.object.active_shape_key_index = key_blocks.find(self._key_block.name)
        elif self.kind == UV:
            layer = self.mesh.uv_layers.get(self.name)
            if layer is not None:
                self.mesh.uv_layers.active = layer
        elif self.kind == COLOR:
            index = self.mesh.color_attributes.find(self.name)
            if index >= 0:
                self.mesh.color_attributes.active_color_index = index
        elif self.kind == WEIGHT:
            group = self.object.vertex_groups.get(self.name)
            if group is not None:
                self.object.vertex_groups.active_index = group.index


def default_target_name(source_name, target_kind):
    """目标名默认沿用源通道名 —— 规则可预期:UVMap → 形态键 'UVMap'。"""
    if target_kind in (POSITION, NORMAL):
        return channel_label(target_kind)
    return source_name or channel_label(target_kind)

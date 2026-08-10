# 通道读取层:把"网格上任意一段数据"统一成 (N, C) float64 + 域(POINT/CORNER),
# 于是匹配基准(Match By)可以拿任意一个通道当两个网格的共同坐标系。
# 覆盖通道:顶点位置 / 形态键 / UV / 自定义法线 / 颜色属性 / 顶点组权重 / 通用属性。
# 位置类通道只交出局部坐标,空间变换一律由会话层统一处理,
# 本层不碰任何矩阵,保证"数值语义"与"坐标空间"两件事互不污染。

import numpy as np

from .correspondence import POINT, CORNER


class ChannelError(RuntimeError):
    """通道读取前提不满足(层不存在、类型不支持等),由会话层转成 ConformError。"""


# ==================== 通道定义 ====================

POSITION = 'POSITION'
SHAPE_KEY = 'SHAPE_KEY'
UV = 'UV'
NORMAL = 'NORMAL'
COLOR = 'COLOR'
WEIGHT = 'WEIGHT'
ATTRIBUTE = 'ATTRIBUTE'

_CHANNEL_LABELS = {
    POSITION: "Vertex Position",
    SHAPE_KEY: "Shape Key",
    UV: "UV Map",
    NORMAL: "Normal",
    COLOR: "Color Attribute",
    WEIGHT: "Vertex Group",
    ATTRIBUTE: "Attribute",
}

# 需要指定层/组/键名的通道(空名 = 用活动项)。
NAMED_CHANNELS = {SHAPE_KEY, UV, COLOR, WEIGHT, ATTRIBUTE}


def channel_label(kind):
    return _CHANNEL_LABELS.get(kind, kind)


# ==================== 源端读取 ====================

class ChannelData:
    """一段读出来的通道:数值 + 域 + 出处。"""

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
    """该通道在这个网格上的全部名字(供面板列举与存在性检查)。"""
    if kind == UV:
        return list(snapshot.uv_layer_names)
    if kind == COLOR:
        return list(snapshot.color_attribute_names)
    if kind == WEIGHT:
        return list(snapshot.vertex_group_names)
    if kind == ATTRIBUTE:
        return list(snapshot.attribute_names)
    if kind == SHAPE_KEY:
        # Basis 是静止态,不算一条形变数据。
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

    side 只影响报错措辞(源侧 / 目标侧),不影响任何行为。
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
            f"used as a matching basis (point and face corner only)")
    return ChannelData(values, domain, name, data_type)

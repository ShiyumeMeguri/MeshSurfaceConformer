# 插件设置:挂在 Scene 上的 PropertyGroup(工具型全局配置,配合"活动物体→选中物体"约定)。
# 分三层暴露,越常用的越靠前:
#   1. 模式(传输 / 转换)+ 匹配预设 —— 面板主区,平时只需要动这两项;
#   2. 数据勾选 / 转换的"从 → 到" —— 主区第二行;
#   3. 高级映射(与 Blender DataTransfer 修改器逐字对齐的顶点域 7 种 + 角点域 6 种,
#      外加本插件独有的 UV Interpolated)、影响、各数据细项 —— 折叠子面板。

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from .channels import (
    ATTRIBUTE,
    COLOR,
    COMPONENT_SOURCE_ITEMS,
    NAMED_CHANNELS,
    NORMAL,
    POSITION,
    SHAPE_KEY,
    SOURCE_CHANNEL_ITEMS,
    TARGET_CHANNEL_ITEMS,
    UV,
    VALUE_REMAP_ITEMS,
    WEIGHT,
)
from .mesh_buffers import ATTRIBUTE_TYPE_ITEMS


def poll_armature(self, candidate):
    return candidate.type == 'ARMATURE'


# Blender rna_enum_dt_method_vertex_items 全集 + UV。
_VERTEX_MAPPING_ITEMS = [
    ('TOPOLOGY', "Topology",
     "Copy from identical topology meshes", 0),
    ('NEAREST', "Nearest Vertex",
     "Copy from closest vertex", 1),
    ('EDGE_NEAREST', "Nearest Edge Vertex",
     "Copy from closest vertex of closest edge", 2),
    ('EDGEINTERP_NEAREST', "Nearest Edge Interpolated",
     "Copy from interpolated values of vertices from closest point on closest edge", 3),
    ('POLY_NEAREST', "Nearest Face Vertex",
     "Copy from closest vertex of closest face", 4),
    ('POLYINTERP_NEAREST', "Nearest Face Interpolated",
     "Copy from interpolated values of vertices from closest point on closest face", 5),
    ('POLYINTERP_VNORPROJ', "Projected Face Interpolated",
     "Copy from interpolated values of vertices from point on closest face hit "
     "by normal-projection", 6),
    ('UV', "UV Interpolated",
     "Match through UV space: sample where each element's UV lands in the source "
     "UV layout (unique to this add-on)", 7),
]

# Blender rna_enum_dt_method_loop_items 全集 + UV。
_CORNER_MAPPING_ITEMS = [
    ('TOPOLOGY', "Topology",
     "Copy from identical topology meshes", 0),
    ('NEAREST_NORMAL', "Nearest Corner and Best Matching Normal",
     "Copy from nearest corner which has the best matching normal", 1),
    ('NEAREST_POLYNOR', "Nearest Corner and Best Matching Face Normal",
     "Copy from nearest corner which has the face with the best matching normal "
     "to destination corner's face one", 2),
    ('NEAREST_POLY', "Nearest Corner of Nearest Face",
     "Copy from nearest corner of nearest face", 3),
    ('POLYINTERP_NEAREST', "Nearest Face Interpolated",
     "Copy from interpolated corners of the nearest source face", 4),
    ('POLYINTERP_LNORPROJ', "Projected Face Interpolated",
     "Copy from interpolated corners of the source face hit by corner normal "
     "projection", 5),
    ('UV', "UV Interpolated",
     "Match through UV space: sample where each corner's UV lands in the source "
     "UV layout (unique to this add-on)", 6),
]

# 匹配基准 = 拿哪一份数据当"两个网格的共同坐标系"。
# 形状基准就是普通的表面贴合;UV 基准就是 UV 空间匹配;颜色/权重/属性同理 ——
# 两边读同一个通道,值相同的地方就是同一个位置,量纲天然对齐无需归一化。
_MATCH_BASIS_ITEMS = [
    (POSITION, "Shape",
     "Match by 3D shape: every target point takes the closest point on the source "
     "surface. The all-round default — works across completely different topologies",
     'MOD_SHRINKWRAP', 0),
    (UV, "UV Map",
     "Match by UV: the target takes whatever the source has at the same UV "
     "coordinate. Two meshes with the same UVs but different shapes line up exactly",
     'UV', 1),
    (COLOR, "Color Attribute",
     "Match by colour: target points take the source point painted with the "
     "same colour", 'COLOR', 2),
    (WEIGHT, "Vertex Group",
     "Match by weight value of one vertex group", 'GROUP_VERTEX', 3),
    (ATTRIBUTE, "Attribute",
     "Match by any point or face-corner attribute — bake your own matching space",
     'MESH_DATA', 4),
    (SHAPE_KEY, "Shape Key",
     "Match by the shape of one shape key instead of the current shape",
     'SHAPEKEY_DATA', 5),
    (NORMAL, "Normal",
     "Match by normal direction", 'NORMALS_VERTEX_FACE', 6),
    ('TOPOLOGY', "Index",
     "Straight index copy. Only valid when both meshes have the same vertex and "
     "corner counts — exact and instant", 'MESH_GRID', 7),
    ('CUSTOM', "Custom",
     "Pick the raw Blender vertex and corner mapping methods yourself in "
     "Advanced Mapping", 'PREFERENCES', 8),
]

_MATCH_METHOD_ITEMS = [
    ('INTERPOLATED', "Interpolated",
     "Take the exact spot between source elements — smooth, and exact when both "
     "meshes share the basis values", 0),
    ('NEAREST', "Nearest",
     "Copy from the single closest source element — values are never blended "
     "(use for hard/discrete data)", 1),
    ('PROJECTED', "Projected",
     "Shoot a ray along each target normal and use where it hits the source "
     "(shape basis only)", 2),
]

# 形状基准的三种方式 → Blender 原生映射标识符(沿用久经验证的老路径)。
_SHAPE_BASIS_VERTEX_MAPPING = {
    'INTERPOLATED': 'POLYINTERP_NEAREST',
    'NEAREST': 'NEAREST',
    'PROJECTED': 'POLYINTERP_VNORPROJ',
}
_SHAPE_BASIS_CORNER_MAPPING = {
    'INTERPOLATED': 'POLYINTERP_NEAREST',
    'NEAREST': 'NEAREST_POLY',
    'PROJECTED': 'POLYINTERP_LNORPROJ',
}

# 需要指定层/组/键名的基准。
NAMED_MATCH_BASES = NAMED_CHANNELS


def resolved_vertex_mapping(settings):
    """面板上的基准 → 引擎实际使用的顶点域映射('BASIS' = 走通用基准路径)。"""
    basis = settings.match_basis
    if basis == 'CUSTOM':
        return settings.vertex_mapping
    if basis == 'TOPOLOGY':
        return 'TOPOLOGY'
    if basis == POSITION:
        return _SHAPE_BASIS_VERTEX_MAPPING[settings.match_method]
    return 'BASIS'


def resolved_corner_mapping(settings):
    """面板上的基准 → 引擎实际使用的角点域映射('BASIS' = 走通用基准路径)。"""
    basis = settings.match_basis
    if basis == 'CUSTOM':
        return settings.corner_mapping
    if basis == 'TOPOLOGY':
        return 'TOPOLOGY'
    if basis == POSITION:
        return _SHAPE_BASIS_CORNER_MAPPING[settings.match_method]
    return 'BASIS'


_ATTRIBUTE_TYPE_ITEMS_WITH_AUTO = [
    ('AUTO', "Automatic",
     "Pick the type from how many components the source data has", 0),
] + [(identifier, label, description, number + 1)
     for identifier, label, description, number in ATTRIBUTE_TYPE_ITEMS]


class MeshSurfaceConformerSettings(PropertyGroup):
    # ---------- 模式 ----------
    mode: EnumProperty(
        name="Mode",
        description="What this panel does",
        items=[
            ('TRANSFER', "Transfer",
             "Copy whole data types from one mesh onto the others "
             "(shape, shape keys, vertex groups, UVs, colors, normals)",
             'MOD_DATA_TRANSFER', 0),
            ('CONVERT', "Convert",
             "Turn any single data channel into any other one — UVs into vertex "
             "positions, positions into UVs, normals into colors, weights into "
             "colors, and so on",
             'ARROW_LEFTRIGHT', 1),
        ],
        default='TRANSFER',
    )

    # ---------- 源 ----------
    use_evaluated_source: BoolProperty(
        name="Use Modified Source",
        description="Sample the source with modifiers and shape keys applied "
                    "(evaluated by the dependency graph)",
        default=False,
    )

    # ---------- 匹配 ----------
    match_basis: EnumProperty(
        name="Match By",
        description="Which data the two meshes are lined up by — any channel can "
                    "act as the shared coordinate system",
        items=_MATCH_BASIS_ITEMS,
        default=POSITION,
    )
    match_method: EnumProperty(
        name="Method",
        description="How a target element picks its spot in the basis space",
        items=_MATCH_METHOD_ITEMS,
        default='INTERPOLATED',
    )
    match_basis_name_source: StringProperty(
        name="Source Layer",
        description="Which layer / group / attribute of the source is the matching "
                    "basis (empty = its active one)",
    )
    match_basis_name_target: StringProperty(
        name="Target Layer",
        description="The matching basis on each target (empty = the layer with the "
                    "same name as the source, otherwise its active one)",
    )
    transform_space: EnumProperty(
        name="Space",
        description="Coordinate space for matching and for mapping positional "
                    "data between the two objects",
        items=[
            ('WORLD', "World", "Match and map positions in world space", 'WORLD', 0),
            ('LOCAL', "Local", "Match and map positions in each object's local space",
             'OBJECT_DATA', 1),
        ],
        default='WORLD',
    )
    vertex_mapping: EnumProperty(
        name="Vertex Mapping",
        description="How vertex data (shape, shape keys, vertex groups, point colors) "
                    "finds its counterpart on the source",
        items=_VERTEX_MAPPING_ITEMS,
        default='POLYINTERP_NEAREST',
    )
    corner_mapping: EnumProperty(
        name="Corner Mapping",
        description="How face corner data (UVs, corner colors, custom normals) "
                    "finds its counterpart on the source",
        items=_CORNER_MAPPING_ITEMS,
        default='POLYINTERP_NEAREST',
    )
    project_bidirectional: BoolProperty(
        name="Bidirectional",
        description="Cast projection rays both along and against the normal and "
                    "keep the nearest hit",
        default=True,
    )
    project_max_distance: FloatProperty(
        name="Ray Length",
        description="Maximum ray distance for normal projection (0 = unlimited)",
        default=0.0,
        min=0.0,
        subtype='DISTANCE',
    )
    corner_sampling_bias: FloatProperty(
        name="Corner Bias",
        description="For interpolated corner mappings: pull the sample point "
                    "slightly towards the face center so corners on either side "
                    "of a seam land on the correct source face",
        default=0.05,
        min=0.001,
        max=0.5,
        precision=3,
    )
    use_max_distance: BoolProperty(
        name="Max Distance",
        description="Ignore matches farther than the distance below, measured in "
                    "the units of whatever Match By basis is active (scene units "
                    "for shape, UV units for UV, 0..1 for colours)",
        default=False,
    )
    max_distance: FloatProperty(
        name="Distance",
        description="Maximum matching distance",
        default=0.1,
        min=0.0,
        subtype='DISTANCE',
    )
    distance_falloff: FloatProperty(
        name="Falloff",
        description="Fade the influence to zero over this range approaching Max "
                    "Distance (0 = hard cutoff)",
        default=0.0,
        min=0.0,
        subtype='DISTANCE',
    )

    # ---------- 影响 ----------
    mix_factor: FloatProperty(
        name="Mix Factor",
        description="Blend between the existing target data and the conformed result",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    vertex_group_mask: StringProperty(
        name="Vertex Group",
        description="Limit the effect to this vertex group on each target "
                    "(resolved by name per target object)",
    )
    invert_vertex_group_mask: BoolProperty(
        name="Invert",
        description="Invert the vertex group mask",
        default=False,
    )
    use_selection_only: BoolProperty(
        name="Only Selected",
        description="Restrict the effect to the vertices selected in Edit Mode",
        default=False,
    )

    # ---------- 数据:形状 ----------
    use_shape: BoolProperty(
        name="Shape",
        description="Conform the target vertex positions onto the source surface",
        default=True,
    )
    shape_as_shape_key: BoolProperty(
        name="As Shape Key",
        description="Write the conformed positions into a shape key instead of "
                    "moving the mesh — the original shape stays recoverable",
        default=True,
    )
    snap_shape_to_vertices: BoolProperty(
        name="Snap to Vertices",
        description="After matching, snap each result to the nearest source vertex "
                    "(off keeps the smooth interpolated surface)",
        default=False,
    )

    # ---------- 数据:形态键 ----------
    use_shape_keys: BoolProperty(
        name="Shape Keys",
        description="Transfer all source shape keys as deltas re-applied on the "
                    "target basis",
        default=False,
    )
    shape_keys_exclude_muted: BoolProperty(
        name="Exclude Muted",
        description="Skip muted shape keys",
        default=False,
    )
    snap_shape_keys_to_vertices: BoolProperty(
        name="Snap to Vertices",
        description="Snap transferred shape key positions to the nearest source "
                    "vertex of that key",
        default=False,
    )
    shape_keys_transfer_drivers: BoolProperty(
        name="Transfer Drivers",
        description="Also copy the drivers of the transferred shape keys "
                    "(uses the armature remap from Rigging Helpers)",
        default=False,
    )

    # ---------- 数据:顶点组 ----------
    use_vertex_groups: BoolProperty(
        name="Vertex Groups",
        description="Transfer all source vertex group weights (interpolated on the "
                    "matched surface)",
        default=False,
    )
    vertex_groups_exclude_locked: BoolProperty(
        name="Exclude Locked",
        description="Skip vertex groups locked on the source",
        default=False,
    )

    # ---------- 数据:UV ----------
    use_uv_layers: BoolProperty(
        name="UVs",
        description="Transfer UV coordinates per face corner (seam-safe: samples "
                    "never bleed across UV islands)",
        default=False,
    )
    uv_transfer_all: BoolProperty(
        name="All Layers",
        description="Transfer every source UV layer instead of a single one",
        default=True,
    )
    uv_transfer_layer_source: StringProperty(
        name="Layer",
        description="Source UV layer to transfer (empty = active layer)",
    )
    uv_write_mode: EnumProperty(
        name="Write To",
        description="Where the transferred UVs are written on the target",
        items=[
            ('MATCH_NAME', "Matching Name",
             "Write into the target layer with the same name, creating it if needed", 0),
            ('ACTIVE', "Active Layer",
             "Write into the target's active UV layer (single-layer transfer only)", 1),
            ('NEW', "New Layer",
             "Always create a new UV layer on the target", 2),
        ],
        default='MATCH_NAME',
    )

    # ---------- 数据:颜色 ----------
    use_color_attributes: BoolProperty(
        name="Color Attributes",
        description="Transfer color attributes (point and face-corner domains)",
        default=False,
    )
    color_transfer_all: BoolProperty(
        name="All Attributes",
        description="Transfer every source color attribute instead of a single one",
        default=True,
    )
    color_transfer_attribute: StringProperty(
        name="Attribute",
        description="Source color attribute to transfer (empty = active)",
    )

    # ---------- 数据:自定义法线 ----------
    use_corner_normals: BoolProperty(
        name="Custom Normals",
        description="Transfer the source's final split normals as custom normals "
                    "on the target",
        default=False,
    )

    # ---------- 转换模式:从 → 到 ----------
    convert_from: EnumProperty(
        name="From",
        description="Which source data to read",
        items=SOURCE_CHANNEL_ITEMS,
        default='UV',
    )
    convert_from_name: StringProperty(
        name="Name",
        description="Which layer / group / key / attribute to read "
                    "(empty = the active one)",
    )
    convert_all_named: BoolProperty(
        name="All, Matched by Name",
        description="Convert every layer / group / shape key of that kind at once, "
                    "pairing source and target by name — this is how whole bone "
                    "weight sets move across in one go",
        default=False,
    )
    convert_to: EnumProperty(
        name="To",
        description="Where the converted values are written",
        items=TARGET_CHANNEL_ITEMS,
        default='SHAPE_KEY',
    )
    convert_to_name: StringProperty(
        name="Name",
        description="Target layer / group / key / attribute name "
                    "(empty = reuse the source name)",
    )
    convert_to_domain: EnumProperty(
        name="Domain",
        description="Domain of a newly created color attribute or attribute",
        items=[
            ('AUTO', "Automatic", "Follow the source data's own domain", 0),
            ('POINT', "Vertex", "One value per vertex", 1),
            ('CORNER', "Face Corner", "One value per face corner", 2),
        ],
        default='AUTO',
    )
    convert_to_attribute_type: EnumProperty(
        name="Type",
        description="Data type of a newly created attribute",
        items=_ATTRIBUTE_TYPE_ITEMS_WITH_AUTO,
        default='AUTO',
    )

    # ---------- 转换模式:数值整形 ----------
    convert_component_mode: EnumProperty(
        name="Components",
        description="How the source components are laid out in the target",
        items=[
            ('AUTO', "Automatic",
             "Same count copies straight across; fewer target components take the "
             "first ones (position to UV = XY); more get padded with 0 and alpha 1 "
             "(UV to position = XY0); a single target component averages the source", 0),
            ('CUSTOM', "Custom",
             "Pick the source of every target component yourself", 1),
        ],
        default='AUTO',
    )
    convert_component_x: EnumProperty(
        name="1st", description="Source of the 1st target component",
        items=COMPONENT_SOURCE_ITEMS, default='X')
    convert_component_y: EnumProperty(
        name="2nd", description="Source of the 2nd target component",
        items=COMPONENT_SOURCE_ITEMS, default='Y')
    convert_component_z: EnumProperty(
        name="3rd", description="Source of the 3rd target component",
        items=COMPONENT_SOURCE_ITEMS, default='Z')
    convert_component_w: EnumProperty(
        name="4th", description="Source of the 4th target component",
        items=COMPONENT_SOURCE_ITEMS, default='ONE')
    convert_value_remap: EnumProperty(
        name="Remap",
        description="Rescale the sampled values before they are laid out",
        items=VALUE_REMAP_ITEMS,
        default='NONE',
    )
    convert_value_scale: FloatProperty(
        name="Scale", description="Multiplier applied to the sampled values",
        default=1.0)
    convert_value_offset: FloatProperty(
        name="Offset", description="Added after the multiplier", default=0.0)

    # ---------- 绑定辅助 ----------
    armature_source: PointerProperty(
        name="Source Armature",
        description="Armature referenced by the source drivers",
        type=bpy.types.Object,
        poll=poll_armature,
    )
    armature_target: PointerProperty(
        name="Target Armature",
        description="Armature that should replace the source armature in the "
                    "copied drivers",
        type=bpy.types.Object,
        poll=poll_armature,
    )


classes = (MeshSurfaceConformerSettings,)

# 插件设置:挂在 Scene 上的 PropertyGroup(工具型全局配置,配合"活动物体→选中物体"约定)。
# 分三层暴露,越常用的越靠前:
#   1. 匹配基准(Match By)+ 方式(Method)—— 面板主区,平时只需要动这两项;
#   2. 六个数据开关 + 四个快捷预设 —— 主区第二行;
#   3. 各数据细项、影响、UV 镜像 —— 折叠子面板。

from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from .channels import (
    ATTRIBUTE,
    COLOR,
    NAMED_CHANNELS,
    NORMAL,
    POSITION,
    SHAPE_KEY,
    UV,
    WEIGHT,
)

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

# 形状基准的三种方式 → 引擎内部的映射标识符(沿用 Blender DataTransfer 的语义与命名)。
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

# 需要投射参数的映射。
PROJECTED_MAPPINGS = {'POLYINTERP_VNORPROJ', 'POLYINTERP_LNORPROJ'}

# 面板枚举 → 内核用的列号:UV 在哪一分量上镜像,结果位置在哪个轴上翻面。
UV_MIRROR_COMPONENTS = {'U': 0, 'V': 1}
RESULT_MIRROR_AXES = {'NONE': None, 'X': 0, 'Y': 1, 'Z': 2}


def resolved_vertex_mapping(settings):
    """面板上的基准 → 引擎实际使用的顶点域映射('BASIS' = 走通用基准路径)。"""
    basis = settings.match_basis
    if basis == 'TOPOLOGY':
        return 'TOPOLOGY'
    if basis == POSITION:
        return _SHAPE_BASIS_VERTEX_MAPPING[settings.match_method]
    return 'BASIS'


def resolved_corner_mapping(settings):
    """面板上的基准 → 引擎实际使用的角点域映射('BASIS' = 走通用基准路径)。"""
    basis = settings.match_basis
    if basis == 'TOPOLOGY':
        return 'TOPOLOGY'
    if basis == POSITION:
        return _SHAPE_BASIS_CORNER_MAPPING[settings.match_method]
    return 'BASIS'


class MeshSurfaceConformerSettings(PropertyGroup):
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
            ('NEW', "New Layer",
             "Always add a new UV layer, numbered .001/.002 — the previous result is "
             "never overwritten, same as how shape keys stack up", 0),
            ('MATCH_NAME', "Matching Name",
             "Overwrite the target layer with the same name, creating it if needed", 1),
            ('ACTIVE', "Active Layer",
             "Overwrite the target's active UV layer (single-layer transfer only)", 2),
        ],
        default='NEW',
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

    # ---------- UV 镜像修复 ----------
    mirror_uv_axis: EnumProperty(
        name="Mirror UV",
        description="Which UV axis the two halves of the layout are mirrored across",
        items=[
            ('U', "U", "The layout is mirrored left/right in UV space", 0),
            ('V', "V", "The layout is mirrored up/down in UV space", 1),
        ],
        default='U',
    )
    mirror_uv_center: FloatProperty(
        name="Center",
        description="UV coordinate the two halves are mirrored about "
                    "(0.5 = the middle of the UV square)",
        default=0.5,
    )
    mirror_result_axis: EnumProperty(
        name="Mirror Positions",
        description="Object axis the sampled positions are mirrored across, in the "
                    "local space of the mesh being fixed",
        items=[
            ('NONE', "None", "Copy the positions across without mirroring them", 0),
            ('X', "X", "Mirror across the local YZ plane", 1),
            ('Y', "Y", "Mirror across the local XZ plane", 2),
            ('Z', "Z", "Mirror across the local XY plane", 3),
        ],
        default='X',
    )
    mirror_source_selection_only: BoolProperty(
        name="Match Selected Source Only",
        description="Only match against the selected part of the source mesh — in "
                    "Edit Mode select the good region on the source and the broken "
                    "region on the mesh being fixed",
        default=False,
    )


classes = (MeshSurfaceConformerSettings,)

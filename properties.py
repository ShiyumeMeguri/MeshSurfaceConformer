# 插件设置:挂在 Object 上的 PropertyGroup(逐对象持久化,迭代工作流友好)。
# UI 文案遵循 Blender 官方英文风格,描述写清语义与单位。

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


def poll_source_mesh(self, candidate):
    """源对象:任意网格,但不能是自己。"""
    return candidate.type == 'MESH' and candidate != bpy.context.active_object


def poll_armature(self, candidate):
    return candidate.type == 'ARMATURE'


class MeshSurfaceConformerSettings(PropertyGroup):
    # ---------- 源 ----------
    source_object: PointerProperty(
        name="Source",
        description="Mesh object to sample data from",
        type=bpy.types.Object,
        poll=poll_source_mesh,
    )
    use_evaluated_source: BoolProperty(
        name="Use Modified Source",
        description="Sample the source with modifiers and shape keys applied "
                    "(evaluated by the dependency graph)",
        default=False,
    )

    # ---------- 匹配 ----------
    matching_domain: EnumProperty(
        name="Match By",
        description="How target elements find their counterpart on the source",
        items=[
            ('SURFACE', "Surface", "Interpolated closest point on the source surface. "
             "Works across completely different topologies", 'SNAP_FACE_CENTER', 0),
            ('UV', "UV", "Match through UV space: elements sample where their UV "
             "lands in the source UV layout", 'UV', 1),
            ('TOPOLOGY', "Topology", "Copy by element index. Requires identical "
             "vertex/corner counts", 'MESH_DATA', 2),
        ],
        default='SURFACE',
    )
    transform_space: EnumProperty(
        name="Space",
        description="Coordinate space for surface matching and for mapping "
                    "positional data between the two objects",
        items=[
            ('WORLD', "World", "Match and map positions in world space", 'WORLD', 0),
            ('LOCAL', "Local", "Match and map positions in each object's local space",
             'OBJECT_DATA', 1),
        ],
        default='WORLD',
    )
    surface_method: EnumProperty(
        name="Method",
        description="How target elements are cast onto the source surface",
        items=[
            ('NEAREST', "Nearest Surface Point",
             "Closest point on the source surface, interpolated inside the hit face",
             'SNAP_ON', 0),
            ('PROJECT', "Normal Projection",
             "Ray cast along the target normals, interpolated inside the hit face",
             'PROP_PROJECTED', 1),
        ],
        default='NEAREST',
    )
    project_bidirectional: BoolProperty(
        name="Bidirectional",
        description="Cast rays both along and against the normal and keep the "
                    "nearest hit",
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
        description="For face-corner data (UVs, colors, normals): pull the sample "
                    "point slightly towards the face center so corners on either "
                    "side of a seam land on the correct source face",
        default=0.05,
        min=0.001,
        max=0.5,
        precision=3,
    )
    uv_match_layer_source: StringProperty(
        name="Source Layer",
        description="Source UV layer used as the matching space "
                    "(empty = active layer)",
    )
    uv_match_layer_target: StringProperty(
        name="Target Layer",
        description="Target UV layer used as the matching space "
                    "(empty = active layer)",
    )
    use_max_distance: BoolProperty(
        name="Max Distance",
        description="Ignore matches farther than the distance below "
                    "(world/local units for surface matching, UV units for UV matching)",
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
        description="Limit the effect to this vertex group on the target",
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
                    "moving the mesh",
        default=False,
    )
    snap_shape_to_vertices: BoolProperty(
        name="Snap to Vertices",
        description="After matching, snap each result to the nearest source vertex "
                    "(legacy behavior — off keeps the smooth interpolated surface)",
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
        default=False,
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
        default=False,
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

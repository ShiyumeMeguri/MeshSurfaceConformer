# UI 层:3D 视图侧边栏(N 面板)主面板 + 子面板。
# 官方布局规范:use_property_split、禁用态灰显、条件显隐;
# 源/目标遵循 Blender 约定(活动物体 = 源,其余选中 = 目标),面板只做状态回显。

import bpy
from bpy.types import Panel

from .operators import gather_source_and_targets

# 需要投射参数的映射。
_PROJECTED_MAPPINGS = {'POLYINTERP_VNORPROJ', 'POLYINTERP_LNORPROJ'}
# 需要角点导向偏置的映射(面角插值系)。
_BIASED_CORNER_MAPPINGS = {'POLYINTERP_NEAREST', 'POLYINTERP_LNORPROJ'}


class ConformerPanelMixin:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # 独立插件用自己的标签页;"Shiyume" 页签仅 ShiyumeTools 专用,严禁挂入。
    bl_category = "Mesh Surface Conformer"

    @staticmethod
    def settings_of(context):
        return context.scene.mesh_surface_conformer


class VIEW3D_PT_mesh_surface_conformer(ConformerPanelMixin, Panel):
    bl_label = "Mesh Surface Conformer"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        source, targets = gather_source_and_targets(context)

        # 源/目标状态回显(Blender 约定:先选目标,最后选源为活动物体)。
        box = layout.box()
        if source is None:
            box.label(text="Active mesh is the source", icon='ERROR')
        else:
            box.label(text=f"Source: {source.name}", icon='OBJECT_DATA')
        if targets:
            if len(targets) == 1:
                box.label(text=f"Target: {targets[0].name}", icon='MOD_DATA_TRANSFER')
            else:
                box.label(text=f"Targets: {len(targets)} selected meshes",
                          icon='MOD_DATA_TRANSFER')
        else:
            box.label(text="Select targets, then the source", icon='INFO')

        layout.prop(settings, "use_evaluated_source")
        layout.separator()
        layout.operator("object.mesh_surface_conform", icon='MOD_DATA_TRANSFER')


class VIEW3D_PT_mesh_surface_conformer_matching(ConformerPanelMixin, Panel):
    bl_label = "Matching"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        source, targets = gather_source_and_targets(context)

        column = layout.column()
        column.prop(settings, "transform_space")
        column.separator()
        column.prop(settings, "vertex_mapping")
        column.prop(settings, "corner_mapping")

        uses_projection = (settings.vertex_mapping in _PROJECTED_MAPPINGS
                           or settings.corner_mapping in _PROJECTED_MAPPINGS)
        if uses_projection:
            column.separator()
            projection_column = column.column()
            projection_column.prop(settings, "project_bidirectional")
            projection_column.prop(settings, "project_max_distance")

        if settings.corner_mapping in _BIASED_CORNER_MAPPINGS:
            column.prop(settings, "corner_sampling_bias")

        if settings.vertex_mapping == 'UV' or settings.corner_mapping == 'UV':
            column.separator()
            if source is not None:
                column.prop_search(
                    settings, "uv_match_layer_source", source.data, "uv_layers",
                    text="Source Layer")
            else:
                column.prop(settings, "uv_match_layer_source", text="Source Layer")
            if len(targets) == 1:
                column.prop_search(
                    settings, "uv_match_layer_target", targets[0].data, "uv_layers",
                    text="Target Layer")
            else:
                column.prop(settings, "uv_match_layer_target", text="Target Layer")

        if settings.vertex_mapping == 'TOPOLOGY' or settings.corner_mapping == 'TOPOLOGY':
            column.separator()
            box = column.box()
            if source is not None and len(targets) == 1:
                source_vertices = len(source.data.vertices)
                target_vertices = len(targets[0].data.vertices)
                box.label(
                    text=f"Vertices: {source_vertices:,} / {target_vertices:,}",
                    icon='CHECKMARK' if source_vertices == target_vertices else 'ERROR')
                source_corners = len(source.data.loops)
                target_corners = len(targets[0].data.loops)
                box.label(
                    text=f"Corners: {source_corners:,} / {target_corners:,}",
                    icon='CHECKMARK' if source_corners == target_corners else 'ERROR')
            else:
                box.label(text="Counts must match per target", icon='INFO')

        if not (settings.vertex_mapping == 'TOPOLOGY'
                and settings.corner_mapping == 'TOPOLOGY'):
            column.separator()
            column.prop(settings, "use_max_distance")
            distance_column = column.column()
            distance_column.active = settings.use_max_distance
            distance_column.prop(settings, "max_distance")
            distance_column.prop(settings, "distance_falloff")


class VIEW3D_PT_mesh_surface_conformer_influence(ConformerPanelMixin, Panel):
    bl_label = "Influence"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        _source, targets = gather_source_and_targets(context)

        column = layout.column()
        column.prop(settings, "mix_factor")

        row = column.row(align=True)
        if len(targets) == 1:
            row.prop_search(
                settings, "vertex_group_mask", targets[0], "vertex_groups")
        else:
            row.prop(settings, "vertex_group_mask")
        invert_row = row.row(align=True)
        invert_row.active = bool(settings.vertex_group_mask)
        invert_row.prop(settings, "invert_vertex_group_mask",
                        text="", icon='ARROW_LEFTRIGHT')

        column.prop(settings, "use_selection_only")


class VIEW3D_PT_mesh_surface_conformer_data(ConformerPanelMixin, Panel):
    bl_label = "Data"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        source, _targets = gather_source_and_targets(context)

        # 形状
        column = layout.column(heading="Shape")
        column.prop(settings, "use_shape", text="Vertex Positions")
        sub_column = column.column()
        sub_column.active = settings.use_shape
        sub_column.prop(settings, "shape_as_shape_key")
        sub_column.prop(settings, "snap_shape_to_vertices")

        # 形态键
        layout.separator()
        column = layout.column(heading="Shape Keys")
        column.prop(settings, "use_shape_keys", text="All Shape Keys")
        sub_column = column.column()
        sub_column.active = settings.use_shape_keys
        sub_column.prop(settings, "shape_keys_exclude_muted")
        sub_column.prop(settings, "snap_shape_keys_to_vertices")
        sub_column.prop(settings, "shape_keys_transfer_drivers")

        # 顶点组
        layout.separator()
        column = layout.column(heading="Vertex Groups")
        column.prop(settings, "use_vertex_groups", text="All Vertex Groups")
        sub_column = column.column()
        sub_column.active = settings.use_vertex_groups
        sub_column.prop(settings, "vertex_groups_exclude_locked")

        # UV
        layout.separator()
        column = layout.column(heading="UVs")
        column.prop(settings, "use_uv_layers", text="UV Layers")
        sub_column = column.column()
        sub_column.active = settings.use_uv_layers
        sub_column.prop(settings, "uv_transfer_all")
        layer_column = sub_column.column()
        layer_column.active = settings.use_uv_layers and not settings.uv_transfer_all
        if source is not None:
            layer_column.prop_search(
                settings, "uv_transfer_layer_source", source.data, "uv_layers")
        else:
            layer_column.prop(settings, "uv_transfer_layer_source")
        sub_column.prop(settings, "uv_write_mode")

        # 颜色属性
        layout.separator()
        column = layout.column(heading="Colors")
        column.prop(settings, "use_color_attributes", text="Color Attributes")
        sub_column = column.column()
        sub_column.active = settings.use_color_attributes
        sub_column.prop(settings, "color_transfer_all")
        attribute_column = sub_column.column()
        attribute_column.active = (settings.use_color_attributes
                                   and not settings.color_transfer_all)
        if source is not None:
            attribute_column.prop_search(
                settings, "color_transfer_attribute", source.data, "color_attributes")
        else:
            attribute_column.prop(settings, "color_transfer_attribute")

        # 自定义法线
        layout.separator()
        column = layout.column(heading="Normals")
        column.prop(settings, "use_corner_normals", text="Custom Normals")


class VIEW3D_PT_mesh_surface_conformer_rigging(ConformerPanelMixin, Panel):
    bl_label = "Rigging Helpers"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 3
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)

        column = layout.column()
        column.prop(settings, "armature_source")
        column.prop(settings, "armature_target")
        layout.separator()
        layout.operator("object.mesh_surface_conform_drivers", icon='DRIVER')


def draw_make_links_menu(self, _context):
    """挂进 Object > Link/Transfer Data 菜单(与官方 Data Transfer 同一入口与同一选择约定)。"""
    self.layout.separator()
    self.layout.operator("object.mesh_surface_conform", icon='MOD_DATA_TRANSFER')


classes = (
    VIEW3D_PT_mesh_surface_conformer,
    VIEW3D_PT_mesh_surface_conformer_matching,
    VIEW3D_PT_mesh_surface_conformer_influence,
    VIEW3D_PT_mesh_surface_conformer_data,
    VIEW3D_PT_mesh_surface_conformer_rigging,
)

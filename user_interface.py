# UI 层:Properties > Object Data 下的主面板 + 子面板。
# 全部遵循官方布局规范:use_property_split、禁用态灰显、条件显隐、图标枚举。

import bpy
from bpy.types import Panel


class ConformerPanelMixin:
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'data'

    @classmethod
    def poll(cls, context):
        active = context.object
        return active is not None and active.type == 'MESH'

    @staticmethod
    def settings_of(context):
        return context.object.mesh_surface_conformer


class DATA_PT_mesh_surface_conformer(ConformerPanelMixin, Panel):
    bl_label = "Mesh Surface Conformer"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)

        column = layout.column()
        column.prop(settings, "source_object")
        column.prop(settings, "use_evaluated_source")

        layout.separator()
        layout.operator("object.mesh_surface_conform", icon='MOD_DATA_TRANSFER')


class DATA_PT_mesh_surface_conformer_matching(ConformerPanelMixin, Panel):
    bl_label = "Matching"
    bl_parent_id = "DATA_PT_mesh_surface_conformer"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        source = settings.source_object

        column = layout.column()
        column.prop(settings, "matching_domain")
        column.prop(settings, "transform_space")

        if settings.matching_domain == 'SURFACE':
            column.separator()
            column.prop(settings, "surface_method")
            if settings.surface_method == 'PROJECT':
                sub_column = column.column()
                sub_column.prop(settings, "project_bidirectional")
                sub_column.prop(settings, "project_max_distance")
            column.prop(settings, "corner_sampling_bias")
        elif settings.matching_domain == 'UV':
            column.separator()
            if source is not None:
                column.prop_search(
                    settings, "uv_match_layer_source", source.data, "uv_layers",
                    text="Source Layer")
            else:
                column.prop(settings, "uv_match_layer_source", text="Source Layer")
            column.prop_search(
                settings, "uv_match_layer_target", context.object.data, "uv_layers",
                text="Target Layer")
        else:
            column.separator()
            box = column.box()
            if source is not None:
                source_vertices = len(source.data.vertices)
                target_vertices = len(context.object.data.vertices)
                matching = source_vertices == target_vertices
                box.label(
                    text=f"Vertices: {source_vertices:,} / {target_vertices:,}",
                    icon='CHECKMARK' if matching else 'ERROR')
                source_corners = len(source.data.loops)
                target_corners = len(context.object.data.loops)
                corners_matching = source_corners == target_corners
                box.label(
                    text=f"Corners: {source_corners:,} / {target_corners:,}",
                    icon='CHECKMARK' if corners_matching else 'ERROR')
            else:
                box.label(text="Pick a source to compare counts", icon='INFO')

        if settings.matching_domain != 'TOPOLOGY':
            column.separator()
            column.prop(settings, "use_max_distance")
            sub_column = column.column()
            sub_column.active = settings.use_max_distance
            sub_column.prop(settings, "max_distance")
            sub_column.prop(settings, "distance_falloff")


class DATA_PT_mesh_surface_conformer_influence(ConformerPanelMixin, Panel):
    bl_label = "Influence"
    bl_parent_id = "DATA_PT_mesh_surface_conformer"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)

        column = layout.column()
        column.prop(settings, "mix_factor")

        row = column.row(align=True)
        row.prop_search(
            settings, "vertex_group_mask", context.object, "vertex_groups")
        invert_row = row.row(align=True)
        invert_row.active = bool(settings.vertex_group_mask)
        invert_row.prop(settings, "invert_vertex_group_mask",
                        text="", icon='ARROW_LEFTRIGHT')

        column.prop(settings, "use_selection_only")


class DATA_PT_mesh_surface_conformer_data(ConformerPanelMixin, Panel):
    bl_label = "Data"
    bl_parent_id = "DATA_PT_mesh_surface_conformer"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        source = settings.source_object

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
        layer_row = sub_column.column()
        layer_row.active = settings.use_uv_layers and not settings.uv_transfer_all
        if source is not None:
            layer_row.prop_search(
                settings, "uv_transfer_layer_source", source.data, "uv_layers")
        else:
            layer_row.prop(settings, "uv_transfer_layer_source")
        sub_column.prop(settings, "uv_write_mode")

        # 颜色属性
        layout.separator()
        column = layout.column(heading="Colors")
        column.prop(settings, "use_color_attributes", text="Color Attributes")
        sub_column = column.column()
        sub_column.active = settings.use_color_attributes
        sub_column.prop(settings, "color_transfer_all")
        attribute_row = sub_column.column()
        attribute_row.active = (settings.use_color_attributes
                                and not settings.color_transfer_all)
        if source is not None:
            attribute_row.prop_search(
                settings, "color_transfer_attribute", source.data, "color_attributes")
        else:
            attribute_row.prop(settings, "color_transfer_attribute")

        # 自定义法线
        layout.separator()
        column = layout.column(heading="Normals")
        column.prop(settings, "use_corner_normals", text="Custom Normals")


class DATA_PT_mesh_surface_conformer_rigging(ConformerPanelMixin, Panel):
    bl_label = "Rigging Helpers"
    bl_parent_id = "DATA_PT_mesh_surface_conformer"
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
    """挂进 Object > Link/Transfer Data 菜单(与官方 Data Transfer 同一入口)。"""
    self.layout.separator()
    self.layout.operator("object.mesh_surface_conform", icon='MOD_DATA_TRANSFER')


classes = (
    DATA_PT_mesh_surface_conformer,
    DATA_PT_mesh_surface_conformer_matching,
    DATA_PT_mesh_surface_conformer_influence,
    DATA_PT_mesh_surface_conformer_data,
    DATA_PT_mesh_surface_conformer_rigging,
)

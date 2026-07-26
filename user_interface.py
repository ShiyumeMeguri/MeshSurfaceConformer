# UI 层:3D 视图侧边栏(N 面板)。
# 设计原则 —— 主面板只放"做什么",细节全部折进子面板:
#   1. 两个大标签页(传输 / 转换)决定形态,一句话状态行讲清"谁 → 谁";
#   2. 一个 Match By 下拉搞定匹配方式(拆开调的人切 Custom 进高级面板);
#   3. 传输 = 快捷预设 + 六个数据开关;转换 = 从 → 到 两行 + 实时分量提示;
#   4. 影响/高级映射/各数据细项默认折叠,平时看不见。

import numpy as np
from bpy.types import Panel

from .channels import (
    ATTRIBUTE,
    COLOR,
    NAMED_CHANNELS,
    NORMAL,
    POSITION,
    SHAPE_KEY,
    UV,
    WEIGHT,
    adapt_components,
    channel_components,
    channel_label,
    channel_names,
    default_target_name,
    resolve_match_name,
    resolve_source_name,
)
from .mesh_buffers import MeshBufferSnapshot, attribute_component_count
from .operators import gather_source_and_targets
from .properties import (
    NAMED_MATCH_BASES,
    resolved_corner_mapping,
    resolved_vertex_mapping,
)


def _mapping_label(mapping, settings):
    """把引擎内部的映射标识符讲成人话。"""
    if mapping == 'BASIS':
        return f"matched by {channel_label(settings.match_basis)}"
    return mapping.replace('_', ' ').title()

# 需要投射参数的映射。
_PROJECTED_MAPPINGS = {'POLYINTERP_VNORPROJ', 'POLYINTERP_LNORPROJ'}
# 需要角点导向偏置的映射(面角插值系)。
_BIASED_CORNER_MAPPINGS = {'POLYINTERP_NEAREST', 'POLYINTERP_LNORPROJ'}

# 通道 → (物体上的属性路径, 集合名),用于 prop_search 直接挑层/组/键。
_CHANNEL_COLLECTIONS = {
    UV: ("data", "uv_layers"),
    COLOR: ("data", "color_attributes"),
    ATTRIBUTE: ("data", "attributes"),
    WEIGHT: (None, "vertex_groups"),
}


def _channel_collection(mesh_object, kind):
    """取出该通道对应的集合宿主与集合名;取不到返回 (None, None)。"""
    if mesh_object is None:
        return None, None
    if kind == SHAPE_KEY:
        shape_keys = mesh_object.data.shape_keys
        return (shape_keys, "key_blocks") if shape_keys is not None else (None, None)
    entry = _CHANNEL_COLLECTIONS.get(kind)
    if entry is None:
        return None, None
    owner_attribute, collection_name = entry
    owner = mesh_object if owner_attribute is None else getattr(
        mesh_object, owner_attribute)
    return owner, collection_name


def _draw_channel_name(layout, settings, property_name, kind, mesh_object, text,
                       placeholder=""):
    """通道名字段:能拿到集合就用搜索框(可挑现成的,也能直接敲新名字);
    拿不到就退成普通输入框,并把留空时的实际取名显示成占位符。"""
    if kind not in NAMED_CHANNELS:
        return
    owner, collection_name = _channel_collection(mesh_object, kind)
    if owner is not None:
        layout.prop_search(settings, property_name, owner, collection_name, text=text)
    else:
        layout.prop(settings, property_name, text=text, placeholder=placeholder)


# 探针值:互不相同且不与常量 0 / 1 撞车,回读时即可认出每个目标分量的来源。
_PROBE_VALUES = (2.0, 20.0, 200.0, 2000.0)
_PROBE_LABELS = {2.0: "X", 20.0: "Y", 200.0: "Z", 2000.0: "W", 0.0: "0", 1.0: "1"}


def _source_component_count(settings, source_object):
    """源通道的分量数;通用属性要看它自己的类型。"""
    kind = settings.convert_from
    components = channel_components(kind)
    if components is not None:
        return components
    if source_object is None:
        return 0
    name = settings.convert_from_name
    attributes = source_object.data.attributes
    attribute = attributes.get(name) if name else attributes.active
    if attribute is None:
        return 0
    return attribute_component_count(attribute.data_type)


def _target_component_count(settings, target_object):
    kind = settings.convert_to
    components = channel_components(kind)
    if components is not None:
        return components
    if target_object is not None:
        name = settings.convert_to_name or settings.convert_from_name
        attribute = target_object.data.attributes.get(name) if name else None
        if attribute is not None:
            return attribute_component_count(attribute.data_type)
    if settings.convert_to_attribute_type != 'AUTO':
        return attribute_component_count(settings.convert_to_attribute_type)
    return _source_component_count(settings, None) or 0


def _resolved_basis_names(settings, source_object, target_object):
    """匹配基准两侧实际会用哪一层 —— 调引擎自己的解析规则,面板不复刻。"""
    basis = settings.match_basis
    source_name = ""
    if source_object is not None:
        source_name = resolve_source_name(
            MeshBufferSnapshot(source_object), basis,
            settings.match_basis_name_source) or ""
    target_name = settings.match_basis_name_target
    if not target_name and target_object is not None:
        target_name = resolve_match_name(
            MeshBufferSnapshot(target_object), basis,
            settings.match_basis_name_target, source_name) or ""
    return source_name, target_name


def _resolved_names(settings, source_object):
    """面板显示的"实际会读/写哪一层" —— 直接调引擎的解析函数,不复刻规则。

    MeshBufferSnapshot 未求值时只是持有物体与网格的空壳,读活动层名不产生任何拷贝。
    """
    if source_object is None:
        return "", ""
    source_name = resolve_source_name(
        MeshBufferSnapshot(source_object), settings.convert_from,
        settings.convert_from_name) or ""
    target_name = settings.convert_to_name or default_target_name(
        source_name, settings.convert_to)
    return source_name, target_name


def _quoted(name):
    return f" '{name}'" if name else ""


def _component_hint(settings, source_components, target_components):
    """把引擎真正会做的分量重排显示出来 —— 直接拿探针跑一遍引擎函数,永不与实现失同步。"""
    if source_components <= 0 or target_components <= 0:
        return ""
    picks = None
    if settings.convert_component_mode == 'CUSTOM':
        picks = [settings.convert_component_x, settings.convert_component_y,
                 settings.convert_component_z, settings.convert_component_w]
    probe = np.array([_PROBE_VALUES[:source_components]], dtype=np.float64)
    result = adapt_components(probe, target_components, picks)[0]
    return " ".join(_PROBE_LABELS.get(round(float(value), 6), "~")
                    for value in result)


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
        settings = self.settings_of(context)
        converting = settings.mode == 'CONVERT'
        source, targets = gather_source_and_targets(context, allow_self=converting)
        in_place = converting and bool(targets) and targets[0] == source

        row = layout.row()
        row.scale_y = 1.2
        row.prop(settings, "mode", expand=True)

        self._draw_status(layout, source, targets, converting, in_place)

        layout.use_property_split = True
        layout.use_property_decorate = False

        if not in_place:
            self._draw_match(layout, settings, source, targets)

        if converting:
            self._draw_convert(layout, settings, source, targets)
        else:
            self._draw_transfer(layout, settings)

        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator(
            "object.mesh_surface_conform",
            text="Convert" if converting else "Transfer",
            icon='ARROW_LEFTRIGHT' if converting else 'MOD_DATA_TRANSFER')

    @staticmethod
    def _draw_status(layout, source, targets, converting, in_place):
        box = layout.box()
        column = box.column(align=True)
        if source is None:
            column.label(text="Make a mesh the active object", icon='ERROR')
            return
        if not targets:
            column.label(text=f"Source: {source.name}", icon='OBJECT_DATA')
            column.label(text="Now select the targets too", icon='INFO')
            return
        if in_place:
            column.label(text=f"In place on: {source.name}", icon='OBJECT_DATA')
            column.label(text="Matched by index — add a 2nd mesh to copy across",
                         icon='MESH_GRID')
            return
        if len(targets) == 1:
            column.label(text=f"{source.name}  →  {targets[0].name}",
                         icon='FORWARD')
        else:
            column.label(text=f"{source.name}  →  {len(targets)} meshes",
                         icon='FORWARD')
        if not converting:
            column.label(text="Active object is the source", icon='INFO')

    @staticmethod
    def _draw_match(layout, settings, source, targets):
        """匹配基准:一个下拉选"拿哪份数据当共同坐标系",需要名字的基准再给两个层选择框。"""
        basis = settings.match_basis
        target_object = targets[0] if len(targets) == 1 else None
        column = layout.column(align=True)
        column.prop(settings, "match_basis")
        if basis in NAMED_MATCH_BASES:
            source_name, target_name = _resolved_basis_names(
                settings, source, target_object)
            _draw_channel_name(column, settings, "match_basis_name_source",
                               basis, source, "Source", source_name)
            _draw_channel_name(column, settings, "match_basis_name_target",
                               basis, target_object, "Target", target_name)
            # 搜索框留空时看不出真正用了哪一层,这里把解析结果摊开说 ——
            # 两边指到不同的层正是"目标被吸成一团"的头号原因。
            if source_name or target_name:
                info = column.column(align=True)
                info.label(
                    text=f"Matching {_quoted(source_name).strip()} → "
                         f"{_quoted(target_name).strip()}",
                    icon='CHECKMARK' if source_name == target_name else 'ERROR')
        if basis not in ('TOPOLOGY', 'CUSTOM'):
            layout.prop(settings, "match_method", text="Method")
            if settings.match_method == 'PROJECTED' and basis != POSITION:
                layout.label(text="Projection needs the Shape basis — "
                                  "interpolating instead", icon='INFO')

    @staticmethod
    def _draw_transfer(layout, settings):
        layout.separator()
        # 一行四个快捷预设,一键配好下面那组开关(占满整行,不与属性标签列争地方)。
        row = layout.row(align=True)
        for preset, text in (('SHAPE', "Shape"), ('WEIGHTS', "Weights"),
                             ('SURFACE', "Detail"), ('ALL', "All")):
            row.operator("object.mesh_surface_conform_data_preset",
                         text=text).preset = preset

        grid = layout.grid_flow(row_major=True, columns=2, even_columns=True,
                                align=True)
        grid.use_property_split = False
        grid.prop(settings, "use_shape", text="Shape", icon='MESH_DATA')
        grid.prop(settings, "use_shape_keys", text="Shape Keys", icon='SHAPEKEY_DATA')
        grid.prop(settings, "use_vertex_groups", text="Vertex Groups",
                  icon='GROUP_VERTEX')
        grid.prop(settings, "use_uv_layers", text="UV Maps", icon='UV')
        grid.prop(settings, "use_color_attributes", text="Colors", icon='COLOR')
        grid.prop(settings, "use_corner_normals", text="Normals",
                  icon='NORMALS_VERTEX_FACE')

    @staticmethod
    def _draw_convert(layout, settings, source, targets):
        layout.separator()
        target_object = targets[0] if targets else None
        source_name, target_name = _resolved_names(settings, source)

        batch = (settings.convert_all_named
                 and settings.convert_from in NAMED_CHANNELS)

        column = layout.column(align=True)
        column.prop(settings, "convert_from", text="From")
        if not batch:
            _draw_channel_name(column, settings, "convert_from_name",
                               settings.convert_from, source, "", source_name)
        if settings.convert_from in NAMED_CHANNELS:
            column.prop(settings, "convert_all_named", text="All, Matched by Name")

        row = layout.row()
        row.alignment = 'CENTER'
        row.operator("object.mesh_surface_conform_swap", text="", icon='FILE_REFRESH',
                     emboss=False)

        column = layout.column(align=True)
        column.prop(settings, "convert_to", text="To")
        if not batch:
            _draw_channel_name(column, settings, "convert_to_name",
                               settings.convert_to, target_object, "", target_name)

        source_components = _source_component_count(settings, source)
        target_components = _target_component_count(settings, target_object)

        box = layout.box()
        column = box.column(align=True)
        if batch:
            count = len(channel_names(MeshBufferSnapshot(source), settings.convert_from)) \
                if source is not None else 0
            column.label(
                text=f"{count} × {channel_label(settings.convert_from)}   →   "
                     f"{channel_label(settings.convert_to)} of the same name",
                icon='FORWARD')
        else:
            column.label(
                text=f"{channel_label(settings.convert_from)}{_quoted(source_name)}"
                     f"   →   {channel_label(settings.convert_to)}"
                     f"{_quoted(target_name)}",
                icon='FORWARD')
        hint = _component_hint(settings, source_components, target_components)
        if hint:
            column.label(
                text=f"{source_components} → {target_components} components:   {hint}",
                icon='NODE_COMPOSITING')


class VIEW3D_PT_mesh_surface_conformer_data(ConformerPanelMixin, Panel):
    bl_label = "Data Options"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 0
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_surface_conformer
        return settings.mode == 'TRANSFER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        source, _targets = gather_source_and_targets(context)
        drawn = False

        if settings.use_shape:
            drawn = True
            column = layout.column(heading="Shape")
            column.prop(settings, "shape_as_shape_key")
            column.prop(settings, "snap_shape_to_vertices")

        if settings.use_shape_keys:
            drawn = True
            layout.separator()
            column = layout.column(heading="Shape Keys")
            column.prop(settings, "shape_keys_exclude_muted")
            column.prop(settings, "snap_shape_keys_to_vertices")
            column.prop(settings, "shape_keys_transfer_drivers")

        if settings.use_vertex_groups:
            drawn = True
            layout.separator()
            column = layout.column(heading="Vertex Groups")
            column.prop(settings, "vertex_groups_exclude_locked")

        if settings.use_uv_layers:
            drawn = True
            layout.separator()
            column = layout.column(heading="UV Maps")
            column.prop(settings, "uv_transfer_all")
            layer_column = column.column()
            layer_column.active = not settings.uv_transfer_all
            if source is not None:
                layer_column.prop_search(
                    settings, "uv_transfer_layer_source", source.data, "uv_layers")
            else:
                layer_column.prop(settings, "uv_transfer_layer_source")
            column.prop(settings, "uv_write_mode")

        if settings.use_color_attributes:
            drawn = True
            layout.separator()
            column = layout.column(heading="Colors")
            column.prop(settings, "color_transfer_all")
            attribute_column = column.column()
            attribute_column.active = not settings.color_transfer_all
            if source is not None:
                attribute_column.prop_search(
                    settings, "color_transfer_attribute", source.data,
                    "color_attributes")
            else:
                attribute_column.prop(settings, "color_transfer_attribute")

        if not drawn:
            layout.label(text="Enable a data type above", icon='INFO')


class VIEW3D_PT_mesh_surface_conformer_convert(ConformerPanelMixin, Panel):
    bl_label = "Convert Options"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 0
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_surface_conformer
        return settings.mode == 'CONVERT'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        _source, targets = gather_source_and_targets(context, allow_self=True)
        target_object = targets[0] if targets else None

        column = layout.column()
        column.prop(settings, "convert_component_mode")
        if settings.convert_component_mode == 'CUSTOM':
            target_components = _target_component_count(settings, target_object)
            component_properties = ("convert_component_x", "convert_component_y",
                                    "convert_component_z", "convert_component_w")
            for index, property_name in enumerate(component_properties):
                if index < target_components:
                    column.prop(settings, property_name)

        layout.separator()
        column = layout.column()
        column.prop(settings, "convert_value_remap")
        if settings.convert_value_remap == 'CUSTOM':
            column.prop(settings, "convert_value_scale")
            column.prop(settings, "convert_value_offset")

        if settings.convert_to in (COLOR, ATTRIBUTE):
            layout.separator()
            column = layout.column()
            column.prop(settings, "convert_to_domain")
            if settings.convert_to == ATTRIBUTE:
                column.prop(settings, "convert_to_attribute_type")
            column.label(text="Only used when creating it", icon='INFO')

        if settings.convert_to in (POSITION, SHAPE_KEY, NORMAL):
            layout.separator()
            box = layout.box()
            box.label(text=f"Values are read as {settings.transform_space.title()} "
                           f"space coordinates", icon='WORLD')


class VIEW3D_PT_mesh_surface_conformer_influence(ConformerPanelMixin, Panel):
    bl_label = "Influence"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 1
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        _source, targets = gather_source_and_targets(context, allow_self=True)

        column = layout.column()
        column.prop(settings, "transform_space")
        column.prop(settings, "use_evaluated_source")

        layout.separator()
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


class VIEW3D_PT_mesh_surface_conformer_matching(ConformerPanelMixin, Panel):
    bl_label = "Advanced Mapping"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 2
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        source, targets = gather_source_and_targets(context)
        if settings.mode == 'CONVERT' and not targets:
            layout.label(text="In-place convert is matched by index", icon='MESH_GRID')
            layout.label(text="Select a second mesh to use these settings", icon='INFO')
            return
        vertex_mapping = resolved_vertex_mapping(settings)
        corner_mapping = resolved_corner_mapping(settings)

        column = layout.column()
        if settings.match_basis == 'CUSTOM':
            column.prop(settings, "vertex_mapping")
            column.prop(settings, "corner_mapping")
        else:
            column.label(text=f"Vertices: {_mapping_label(vertex_mapping, settings)}")
            column.label(text=f"Corners: {_mapping_label(corner_mapping, settings)}")
            column.label(text="Switch Match By to Custom to edit", icon='INFO')

        if (vertex_mapping in _PROJECTED_MAPPINGS
                or corner_mapping in _PROJECTED_MAPPINGS):
            column.separator()
            projection_column = column.column()
            projection_column.prop(settings, "project_bidirectional")
            projection_column.prop(settings, "project_max_distance")

        if corner_mapping in _BIASED_CORNER_MAPPINGS:
            column.prop(settings, "corner_sampling_bias")

        if vertex_mapping == 'UV' or corner_mapping == 'UV':
            column.separator()
            _draw_channel_name(column, settings, "match_basis_name_source",
                               UV, source, "Source Layer")
            _draw_channel_name(column, settings, "match_basis_name_target", UV,
                               targets[0] if len(targets) == 1 else None,
                               "Target Layer", settings.match_basis_name_source)

        if vertex_mapping == 'TOPOLOGY' or corner_mapping == 'TOPOLOGY':
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

        if not (vertex_mapping == 'TOPOLOGY' and corner_mapping == 'TOPOLOGY'):
            column.separator()
            column.prop(settings, "use_max_distance")
            distance_column = column.column()
            distance_column.active = settings.use_max_distance
            distance_column.prop(settings, "max_distance")
            distance_column.prop(settings, "distance_falloff")


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
    VIEW3D_PT_mesh_surface_conformer_data,
    VIEW3D_PT_mesh_surface_conformer_convert,
    VIEW3D_PT_mesh_surface_conformer_influence,
    VIEW3D_PT_mesh_surface_conformer_matching,
    VIEW3D_PT_mesh_surface_conformer_rigging,
)

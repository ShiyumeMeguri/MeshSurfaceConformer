# UI 层:3D 视图侧边栏(N 面板)。
# 设计原则 —— 主面板只放"做什么",细节全部折进子面板:
#   1. 一句话状态行讲清"谁 → 谁";
#   2. 一个 Match By 下拉搞定匹配基准,需要指定层的基准跟出两个层选择框;
#   3. 快捷预设 + 六个数据开关 → Transfer;
#   4. 各数据细项 / 影响 / UV 镜像折进子面板。

from bpy.types import Panel

from .channels import (
    ATTRIBUTE,
    COLOR,
    NAMED_CHANNELS,
    POSITION,
    SHAPE_KEY,
    UV,
    WEIGHT,
    channel_label,
    channel_names,
    resolve_match_name,
    resolve_source_name,
)
from .mesh_buffers import MeshBufferSnapshot
from .operators import gather_source_and_target, other_selected_meshes
from .properties import (
    NAMED_MATCH_BASES,
    PROJECTED_MAPPINGS,
    resolved_corner_mapping,
    resolved_vertex_mapping,
)

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


def _resolved_basis_names(settings, source_object, target_object):
    """匹配基准两侧实际会用哪一层 —— 调引擎自己的解析规则,面板不复刻。

    MeshBufferSnapshot 未求值时只是持有物体与网格的空壳,读活动层名不产生任何拷贝。
    """
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


def _quoted(name):
    return f" '{name}'" if name else ""


def _channel_exists(mesh_object, kind, name):
    """该物体上有没有这个名字的层/组/键 —— 唯一值得报错的情况。
    两边指到不同名字的层是正常用法(以任意一层为基准匹配另一层),不是错误。"""
    if mesh_object is None or not name:
        return True
    snapshot = MeshBufferSnapshot(mesh_object)
    if kind == SHAPE_KEY:
        return name in snapshot.shape_key_names
    return name in channel_names(snapshot, kind)


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
        source, target = gather_source_and_target(context)

        self._draw_status(layout, context, source, target)

        layout.use_property_split = True
        layout.use_property_decorate = False
        self._draw_match(layout, settings, source, target)
        self._draw_data(layout, settings)

        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("object.mesh_surface_conform", text="Transfer",
                     icon='MOD_DATA_TRANSFER')

    @staticmethod
    def _draw_status(layout, context, source, target):
        box = layout.box()
        column = box.column(align=True)
        active = context.active_object
        if active is None or active.type != 'MESH':
            column.label(text="Make the mesh you want to modify the active object",
                         icon='ERROR')
            return
        if source is None:
            others = other_selected_meshes(context, active)
            if len(others) > 1:
                column.label(text=f"{len(others)} possible sources selected",
                             icon='ERROR')
                column.label(text="Select exactly one source plus the mesh to modify",
                             icon='INFO')
            else:
                column.label(text=f"Modifying: {active.name}", icon='OBJECT_DATA')
                column.label(text="Now also select the source mesh", icon='INFO')
            return
        column.label(text=f"{source.name}  →  {target.name}", icon='FORWARD')
        column.label(text="Active object is the one being modified", icon='INFO')

    @staticmethod
    def _draw_match(layout, settings, source, target):
        """匹配基准:一个下拉选"拿哪份数据当共同坐标系",需要名字的基准再给两个层选择框。"""
        basis = settings.match_basis
        column = layout.column(align=True)
        column.prop(settings, "match_basis")
        if basis in NAMED_MATCH_BASES:
            source_name, target_name = _resolved_basis_names(settings, source, target)
            _draw_channel_name(column, settings, "match_basis_name_source",
                               basis, source, "Source", source_name)
            _draw_channel_name(column, settings, "match_basis_name_target",
                               basis, target, "Target", target_name)
            # 搜索框留空时看不出真正用了哪一层,这里把解析结果摊开说。
            # 只有"这一层根本不存在"才是错误;名字不同是正常的跨层匹配。
            if source_name or target_name:
                source_missing = not _channel_exists(source, basis, source_name)
                target_missing = not _channel_exists(target, basis, target_name)
                info = column.column(align=True)
                if source_missing or target_missing:
                    side = ("source and target" if source_missing and target_missing
                            else "source" if source_missing else "target")
                    info.label(
                        text=f"The {side} has no "
                             f"{channel_label(basis).lower()} by that name",
                        icon='ERROR')
                else:
                    info.label(
                        text=f"Matching {_quoted(source_name).strip()} → "
                             f"{_quoted(target_name).strip()}",
                        icon='CHECKMARK')
        if basis == 'TOPOLOGY':
            box = layout.box()
            if source is not None and target is not None:
                source_vertices = len(source.data.vertices)
                target_vertices = len(target.data.vertices)
                box.label(
                    text=f"Vertices: {source_vertices:,} / {target_vertices:,}",
                    icon='CHECKMARK' if source_vertices == target_vertices else 'ERROR')
                source_corners = len(source.data.loops)
                target_corners = len(target.data.loops)
                box.label(
                    text=f"Corners: {source_corners:,} / {target_corners:,}",
                    icon='CHECKMARK' if source_corners == target_corners else 'ERROR')
            else:
                box.label(text="Vertex and corner counts must match", icon='INFO')
            return
        layout.prop(settings, "match_method", text="Method")
        if settings.match_method == 'PROJECTED' and basis != POSITION:
            layout.label(text="Projection needs the Shape basis — "
                              "interpolating instead", icon='INFO')
        elif resolved_vertex_mapping(settings) in PROJECTED_MAPPINGS \
                or resolved_corner_mapping(settings) in PROJECTED_MAPPINGS:
            column = layout.column(align=True)
            column.prop(settings, "project_bidirectional")
            column.prop(settings, "project_max_distance")

    @staticmethod
    def _draw_data(layout, settings):
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


class VIEW3D_PT_mesh_surface_conformer_data(ConformerPanelMixin, Panel):
    bl_label = "Data Options"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 0
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = self.settings_of(context)
        source, _target = gather_source_and_target(context)
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
            column.prop(settings, "shape_keys_transfer_all")
            key_column = column.column()
            key_column.active = not settings.shape_keys_transfer_all
            source_keys = source.data.shape_keys if source is not None else None
            if source_keys is not None:
                key_column.prop_search(
                    settings, "shape_keys_transfer_key", source_keys, "key_blocks")
            else:
                key_column.prop(settings, "shape_keys_transfer_key")
            column.prop(settings, "shape_keys_exclude_muted")
            column.prop(settings, "snap_shape_keys_to_vertices")

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
        _source, target = gather_source_and_target(context, allow_self=True)

        column = layout.column()
        column.prop(settings, "transform_space")
        column.prop(settings, "use_evaluated_source")

        layout.separator()
        column = layout.column()
        column.prop(settings, "mix_factor")

        row = column.row(align=True)
        if target is not None:
            row.prop_search(settings, "vertex_group_mask", target, "vertex_groups")
        else:
            row.prop(settings, "vertex_group_mask")
        invert_row = row.row(align=True)
        invert_row.active = bool(settings.vertex_group_mask)
        invert_row.prop(settings, "invert_vertex_group_mask",
                        text="", icon='ARROW_LEFTRIGHT')

        column.prop(settings, "use_selection_only")

        if settings.match_basis != 'TOPOLOGY':
            layout.separator()
            column = layout.column()
            column.prop(settings, "use_max_distance")
            distance_column = column.column()
            distance_column.active = settings.use_max_distance
            distance_column.prop(settings, "max_distance")
            distance_column.prop(settings, "distance_falloff")


class VIEW3D_PT_mesh_surface_conformer_mirror(ConformerPanelMixin, Panel):
    bl_label = "UV Mirror"
    bl_parent_id = "VIEW3D_PT_mesh_surface_conformer"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        settings = self.settings_of(context)
        source, target = gather_source_and_target(context, allow_self=True)
        active = context.active_object
        editing = active is not None and active.mode == 'EDIT'

        box = layout.box()
        column = box.column(align=True)
        if source is not None and source == target:
            column.label(text=f"Mirrored onto itself: {target.name}",
                         icon='MOD_MIRROR')
        elif source is not None:
            column.label(text=f"{source.name}  →  {target.name}", icon='FORWARD')
        column.label(
            text="Fixes the selected vertices" if editing else "Fixes every vertex",
            icon='EDITMODE_HLT' if editing else 'OBJECT_DATAMODE')
        column.label(text="Matched through the active UV map on both meshes",
                     icon='UV')

        layout.use_property_split = True
        layout.use_property_decorate = False
        column = layout.column(align=True)
        column.prop(settings, "mirror_uv_axis")
        column.prop(settings, "mirror_uv_center")
        layout.prop(settings, "mirror_result_axis")
        layout.prop(settings, "mirror_source_selection_only")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("object.mesh_surface_conform_uv_mirror", icon='MOD_MIRROR')


def draw_make_links_menu(self, _context):
    """挂进 Object > Link/Transfer Data 菜单(与官方 Data Transfer 同一入口与同一选择约定)。"""
    self.layout.separator()
    self.layout.operator("object.mesh_surface_conform", icon='MOD_DATA_TRANSFER')


def draw_edit_mesh_vertex_menu(self, _context):
    """挂进编辑模式的 Vertex 菜单(Ctrl+V):选中顶点 → 按 UV 镜像修坐标。"""
    self.layout.separator()
    self.layout.operator("object.mesh_surface_conform_uv_mirror", icon='MOD_MIRROR')


classes = (
    VIEW3D_PT_mesh_surface_conformer,
    VIEW3D_PT_mesh_surface_conformer_data,
    VIEW3D_PT_mesh_surface_conformer_influence,
    VIEW3D_PT_mesh_surface_conformer_mirror,
)

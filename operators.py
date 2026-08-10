# 算子层:活动物体 = 被修改的目标,另一个选中的网格 = 源
# (最后选中的就是要动的那个);设置读自 Scene 级 PropertyGroup。
# UV 镜像修复额外允许"只选一个物体"= 拿自己 UV 的另一半修自己。
# 引擎全程纯数据操作(无 bpy.ops),唯一的模式切换用于把编辑模式选择/几何冲刷回网格。

import time

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator

from .conform_session import ConformSession, ConformError, MirrorPlan
from .properties import RESULT_MIRROR_AXES, UV_MIRROR_COMPONENTS

# 快捷预设 → 六个数据开关的取值(顺序:形状/形态键/顶点组/UV/颜色/法线)。
_DATA_PRESETS = {
    'SHAPE': (True, False, False, False, False, False),
    'WEIGHTS': (False, False, True, False, False, False),
    'SURFACE': (False, False, False, True, True, True),
    'ALL': (True, True, True, True, True, True),
    'NONE': (False, False, False, False, False, False),
}

_DATA_TOGGLES = ("use_shape", "use_shape_keys", "use_vertex_groups",
                 "use_uv_layers", "use_color_attributes", "use_corner_normals")


def _any_data_type_enabled(settings):
    return any(getattr(settings, name) for name in _DATA_TOGGLES)


def other_selected_meshes(context, active):
    return [candidate for candidate in context.selected_objects
            if candidate.type == 'MESH' and candidate != active]


def gather_source_and_target(context, allow_self=False):
    """活动物体 = 被修改的目标,另一个选中的网格 = 源。

    最后选中的(活动物体)就是要动的那个 —— 与 Ctrl+L / object.data_transfer 相反,
    那套约定把活动物体当源,读起来是反的。
    源必须唯一:Blender 不保留选择顺序,选中三个以上就无从判断谁是源。
    allow_self=True 且没有别的选中网格时,源就是自己。
    """
    target = context.active_object
    if target is None or target.type != 'MESH':
        return None, None
    sources = other_selected_meshes(context, target)
    if not sources:
        return (target, target) if allow_self else (None, None)
    if len(sources) > 1:
        return None, None
    return sources[0], target


def poll_selection(cls, context, allow_self):
    """两个算子共用的选择前提。"""
    active = context.active_object
    if active is None or active.type != 'MESH':
        cls.poll_message_set("Make the mesh you want to modify the active object")
        return False
    if len(other_selected_meshes(context, active)) > 1:
        cls.poll_message_set("Select exactly one source plus the mesh to modify")
        return False
    source, _target = gather_source_and_target(context, allow_self)
    if source is None:
        cls.poll_message_set("Select the source first, then the mesh to modify")
        return False
    return True


def run_session(operator, context, settings, source, target, verb, mirror=None):
    """跑一次会话并把结果/警告报给用户。编辑模式下先冲刷回网格,收尾再切回去。"""
    started_at = time.perf_counter()
    # 编辑网格与选择状态必须先冲刷(多物体编辑一并退出);数据级写回只在物体模式成立。
    # 判据取活动物体(= 目标)的模式:源可能根本不在编辑模式里。
    original_mode = context.active_object.mode
    if original_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    session = None
    try:
        session = ConformSession(context, settings, source, target, mirror)
        summaries, warnings = session.run()
    except ConformError as error:
        if session is not None:
            for warning in session.warnings:
                operator.report({'WARNING'}, f"{target.name}: {warning}")
        operator.report({'ERROR'}, f"{target.name}: {error}")
        return {'CANCELLED'}
    finally:
        if session is not None:
            session.free()
        if original_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=original_mode)

    for warning in warnings:
        operator.report({'WARNING'}, f"{target.name}: {warning}")
    elapsed = time.perf_counter() - started_at
    detail = ", ".join(summaries)
    if source == target:
        operator.report(
            {'INFO'}, f"{verb} {detail} on '{target.name}' in {elapsed:.2f}s")
    else:
        operator.report(
            {'INFO'},
            f"{verb} {detail} from '{source.name}' to '{target.name}' "
            f"in {elapsed:.2f}s")
    return {'FINISHED'}


class OBJECT_OT_mesh_surface_conform(Operator):
    """Transfer the enabled data types from the selected source mesh onto the active mesh"""
    bl_idname = "object.mesh_surface_conform"
    bl_label = "Conform Surface Data"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not poll_selection(cls, context, allow_self=False):
            return False
        if not _any_data_type_enabled(context.scene.mesh_surface_conformer):
            cls.poll_message_set("Enable at least one data type")
            return False
        return True

    def execute(self, context):
        settings = context.scene.mesh_surface_conformer
        source, target = gather_source_and_target(context)
        if source is None:
            self.report({'ERROR'}, "Select the source first, then the mesh to modify")
            return {'CANCELLED'}
        return run_session(self, context, settings, source, target, "Conformed")


class OBJECT_OT_mesh_surface_conform_data_preset(Operator):
    """Turn on just the data types needed for this job"""
    bl_idname = "object.mesh_surface_conform_data_preset"
    bl_label = "Data Preset"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    preset: EnumProperty(
        name="Preset",
        items=[
            ('SHAPE', "Shape", "Vertex positions only", 0),
            ('WEIGHTS', "Weights", "Vertex groups only", 1),
            ('SURFACE', "Surface Detail", "UVs, color attributes and custom normals", 2),
            ('ALL', "Everything", "Every data type at once", 3),
            ('NONE', "Clear", "Turn everything off", 4),
        ],
        default='SHAPE',
    )

    def execute(self, context):
        settings = context.scene.mesh_surface_conformer
        for name, value in zip(_DATA_TOGGLES, _DATA_PRESETS[self.preset]):
            setattr(settings, name, value)
        return {'FINISHED'}


class OBJECT_OT_mesh_surface_conform_uv_mirror(Operator):
    """Fix vertex positions from their UV-mirrored counterparts — on this mesh, or copied from another one"""
    bl_idname = "object.mesh_surface_conform_uv_mirror"
    bl_label = "Fix by UV Mirror"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not poll_selection(cls, context, allow_self=True):
            return False
        source, target = gather_source_and_target(context, allow_self=True)
        for mesh_object in ({source, target}):
            if not mesh_object.data.uv_layers:
                cls.poll_message_set(f"'{mesh_object.name}' has no UV map")
                return False
        return True

    def execute(self, context):
        settings = context.scene.mesh_surface_conformer
        source, target = gather_source_and_target(context, allow_self=True)
        if source is None:
            self.report({'ERROR'}, "Select the source first, then the mesh to fix")
            return {'CANCELLED'}
        mirror = MirrorPlan(
            basis_component=UV_MIRROR_COMPONENTS[settings.mirror_uv_axis],
            basis_center=settings.mirror_uv_center,
            result_axis=RESULT_MIRROR_AXES[settings.mirror_result_axis],
            source_selection_only=settings.mirror_source_selection_only,
            # 编辑模式下就该只修选中的顶点 —— 那正是这个功能的用法。
            selection_only=context.active_object.mode == 'EDIT')
        return run_session(self, context, settings, source, target, "Mirrored", mirror)


classes = (
    OBJECT_OT_mesh_surface_conform,
    OBJECT_OT_mesh_surface_conform_data_preset,
    OBJECT_OT_mesh_surface_conform_uv_mirror,
)

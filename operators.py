# 算子层:遵循 Blender 官方 Link/Transfer Data 约定 —— 活动物体是源,
# 其余选中网格是目标,无需手动指定源;设置读自 Scene 级 PropertyGroup。
# 转换模式额外允许"只选一个物体"= 就地把自己的一个通道转成另一个通道。
# 引擎全程纯数据操作(无 bpy.ops),唯一的模式切换用于把编辑模式选择/几何冲刷回网格。

import time

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator

from .conform_session import ConformSession, ConformError
from .shape_key_drivers import transfer_shape_key_drivers

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


def gather_source_and_targets(context, allow_self=False):
    """活动物体 = 源,其余选中网格 = 目标(与 object.data_transfer 同一约定)。

    allow_self=True 且没有别的选中网格时,目标就是源自己(就地转换)。
    """
    source = context.active_object
    if source is None or source.type != 'MESH':
        return None, []
    targets = [candidate for candidate in context.selected_objects
               if candidate.type == 'MESH' and candidate != source]
    if not targets and allow_self:
        targets = [source]
    return source, targets


class OBJECT_OT_mesh_surface_conform(Operator):
    """Transfer the enabled data types from the active mesh onto the other selected meshes, or convert one data channel into another"""
    bl_idname = "object.mesh_surface_conform"
    bl_label = "Conform Surface Data"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_surface_conformer
        converting = settings.mode == 'CONVERT'
        source, targets = gather_source_and_targets(context, allow_self=converting)
        if source is None:
            cls.poll_message_set("Active object must be a mesh")
            return False
        if not targets:
            cls.poll_message_set("Select the target meshes, then the source last")
            return False
        if not converting and not _any_data_type_enabled(settings):
            cls.poll_message_set("Enable at least one data type")
            return False
        return True

    def execute(self, context):
        settings = context.scene.mesh_surface_conformer
        converting = settings.mode == 'CONVERT'
        source, targets = gather_source_and_targets(context, allow_self=converting)
        started_at = time.perf_counter()

        original_mode = source.mode
        if original_mode != 'OBJECT':
            # 冲刷编辑网格与选择状态(多物体编辑一并退出);数据级写回必须在物体模式。
            bpy.ops.object.mode_set(mode='OBJECT')

        conformed_targets = []
        last_summaries = []
        try:
            for target in targets:
                session = ConformSession(context, settings, source, target)
                try:
                    summaries, warnings = session.run()
                except ConformError as error:
                    for warning in session.warnings:
                        self.report({'WARNING'}, f"{target.name}: {warning}")
                    self.report({'ERROR'}, f"{target.name}: {error}")
                    continue
                finally:
                    session.free()
                for warning in warnings:
                    self.report({'WARNING'}, f"{target.name}: {warning}")
                conformed_targets.append(target.name)
                last_summaries = summaries
        finally:
            if original_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=original_mode)

        if not conformed_targets:
            return {'CANCELLED'}
        elapsed = time.perf_counter() - started_at
        verb = "Converted" if converting else "Conformed"
        if len(conformed_targets) == 1:
            if converting and source == targets[0]:
                self.report(
                    {'INFO'},
                    f"{verb} {', '.join(last_summaries)} on '{conformed_targets[0]}' "
                    f"in {elapsed:.2f}s")
            else:
                self.report(
                    {'INFO'},
                    f"{verb} {', '.join(last_summaries)} from '{source.name}' to "
                    f"'{conformed_targets[0]}' in {elapsed:.2f}s")
        else:
            self.report(
                {'INFO'},
                f"{verb} {len(conformed_targets)} objects from '{source.name}' "
                f"in {elapsed:.2f}s")
        return {'FINISHED'}


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


class OBJECT_OT_mesh_surface_conform_swap(Operator):
    """Swap the From and To channels"""
    bl_idname = "object.mesh_surface_conform_swap"
    bl_label = "Swap Direction"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        settings = context.scene.mesh_surface_conformer
        source_kind, source_name = settings.convert_from, settings.convert_from_name
        settings.convert_from = settings.convert_to
        settings.convert_from_name = settings.convert_to_name
        settings.convert_to = source_kind
        settings.convert_to_name = source_name
        return {'FINISHED'}


class OBJECT_OT_mesh_surface_conform_drivers(Operator):
    """Copy the shape key drivers from the active mesh to the matching shape keys on the other selected meshes"""
    bl_idname = "object.mesh_surface_conform_drivers"
    bl_label = "Transfer Shape Key Drivers"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        source, targets = gather_source_and_targets(context)
        if source is None:
            cls.poll_message_set("Active object must be a mesh (the source)")
            return False
        if not targets:
            cls.poll_message_set("Select the target meshes, then the source last")
            return False
        if source.data.shape_keys is None:
            cls.poll_message_set("Source has no shape keys")
            return False
        return True

    def execute(self, context):
        source, targets = gather_source_and_targets(context)
        settings = context.scene.mesh_surface_conformer
        transferred_total = 0
        for target in targets:
            if target.data.shape_keys is None:
                self.report(
                    {'WARNING'},
                    f"{target.name}: no shape keys — transfer shape keys first")
                continue
            transferred_total += transfer_shape_key_drivers(
                source, target, settings.armature_source, settings.armature_target)
        if transferred_total == 0:
            self.report({'WARNING'}, "No matching shape key drivers found")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Transferred {transferred_total} shape key driver(s)")
        return {'FINISHED'}


classes = (
    OBJECT_OT_mesh_surface_conform,
    OBJECT_OT_mesh_surface_conform_data_preset,
    OBJECT_OT_mesh_surface_conform_swap,
    OBJECT_OT_mesh_surface_conform_drivers,
)

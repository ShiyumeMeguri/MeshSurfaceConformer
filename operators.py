# 算子层:读取对象上的设置,驱动 ConformSession,汇报结果。
# 引擎全程纯数据操作(无 bpy.ops),唯一的模式切换用于把编辑模式选择/几何冲刷回网格。

import time

import bpy
from bpy.types import Operator

from .conform_session import ConformSession, ConformError
from .shape_key_drivers import transfer_shape_key_drivers


def _any_data_type_enabled(settings):
    return (settings.use_shape or settings.use_shape_keys or settings.use_vertex_groups
            or settings.use_uv_layers or settings.use_color_attributes
            or settings.use_corner_normals)


class OBJECT_OT_mesh_surface_conform(Operator):
    """Conform the enabled data types of the active mesh onto the source surface"""
    bl_idname = "object.mesh_surface_conform"
    bl_label = "Conform Surface Data"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        active = context.active_object
        if active is None or active.type != 'MESH':
            cls.poll_message_set("Active object must be a mesh")
            return False
        settings = active.mesh_surface_conformer
        source = settings.source_object
        if source is None or source.type != 'MESH' or source == active:
            cls.poll_message_set("Pick a different mesh as the source")
            return False
        if not _any_data_type_enabled(settings):
            cls.poll_message_set("Enable at least one data type")
            return False
        return True

    def execute(self, context):
        target_object = context.active_object
        settings = target_object.mesh_surface_conformer
        started_at = time.perf_counter()

        original_mode = target_object.mode
        if original_mode != 'OBJECT':
            # 冲刷编辑网格与选择状态;数据级写回也必须在物体模式下进行。
            bpy.ops.object.mode_set(mode='OBJECT')

        session = None
        try:
            session = ConformSession(context, settings, target_object)
            summaries, warnings = session.run()
        except ConformError as error:
            if session is not None:
                for warning in session.warnings:
                    self.report({'WARNING'}, warning)
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            if session is not None:
                session.free()
            if original_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=original_mode)

        for warning in warnings:
            self.report({'WARNING'}, warning)
        elapsed = time.perf_counter() - started_at
        self.report(
            {'INFO'},
            f"Conformed {', '.join(summaries)} from '{settings.source_object.name}' "
            f"in {elapsed:.2f}s")
        return {'FINISHED'}


class OBJECT_OT_mesh_surface_conform_drivers(Operator):
    """Copy the shape key drivers from the source to the matching shape keys on the active mesh"""
    bl_idname = "object.mesh_surface_conform_drivers"
    bl_label = "Transfer Shape Key Drivers"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        active = context.active_object
        if active is None or active.type != 'MESH':
            cls.poll_message_set("Active object must be a mesh")
            return False
        settings = active.mesh_surface_conformer
        source = settings.source_object
        if source is None or source.type != 'MESH' or source == active:
            cls.poll_message_set("Pick a different mesh as the source")
            return False
        if active.data.shape_keys is None:
            cls.poll_message_set("Target has no shape keys — transfer shape keys first")
            return False
        return True

    def execute(self, context):
        target_object = context.active_object
        settings = target_object.mesh_surface_conformer
        transferred_count = transfer_shape_key_drivers(
            settings.source_object, target_object,
            settings.armature_source, settings.armature_target)
        if transferred_count == 0:
            self.report({'WARNING'}, "No matching shape key drivers found on the source")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Transferred {transferred_count} shape key driver(s)")
        return {'FINISHED'}


classes = (
    OBJECT_OT_mesh_surface_conform,
    OBJECT_OT_mesh_surface_conform_drivers,
)

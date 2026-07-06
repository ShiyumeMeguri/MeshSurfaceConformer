# 算子层:遵循 Blender 官方 Link/Transfer Data 约定 —— 活动物体是源,
# 其余选中网格是目标,无需手动指定源;设置读自 Scene 级 PropertyGroup。
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


def gather_source_and_targets(context):
    """活动物体 = 源,其余选中网格 = 目标(与 object.data_transfer 同一约定)。"""
    source = context.active_object
    if source is None or source.type != 'MESH':
        return None, []
    targets = [candidate for candidate in context.selected_objects
               if candidate.type == 'MESH' and candidate != source]
    return source, targets


class OBJECT_OT_mesh_surface_conform(Operator):
    """Conform the enabled data types from the active mesh onto the other selected meshes"""
    bl_idname = "object.mesh_surface_conform"
    bl_label = "Conform Surface Data"
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
        if not _any_data_type_enabled(context.scene.mesh_surface_conformer):
            cls.poll_message_set("Enable at least one data type")
            return False
        return True

    def execute(self, context):
        source, targets = gather_source_and_targets(context)
        settings = context.scene.mesh_surface_conformer
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
        if len(conformed_targets) == 1:
            self.report(
                {'INFO'},
                f"Conformed {', '.join(last_summaries)} from '{source.name}' to "
                f"'{conformed_targets[0]}' in {elapsed:.2f}s")
        else:
            self.report(
                {'INFO'},
                f"Conformed {len(conformed_targets)} objects from '{source.name}' "
                f"in {elapsed:.2f}s")
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
    OBJECT_OT_mesh_surface_conform_drivers,
)

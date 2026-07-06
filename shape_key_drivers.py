# 形态键驱动器复制单元(自旧版 MeshDataTransfer.transfer_shape_keys_drivers 重构)。
# 修复点:
#   1. 目标已有同通道驱动器时先移除再复制,不再叠加出重复 F-Curve;
#   2. 变量目标属性用统一守卫复制,兼容 SINGLE_PROP / TRANSFORMS / ROTATION_DIFF /
#      LOC_DIFF 以及新增的 CONTEXT_PROP 等类型,不再按类型手抄三份分支;
#   3. id_type 只在可写时设置(非 SINGLE_PROP 变量该属性只读)。

import codecs


def _extract_shape_key_name(data_path):
    """从 'key_blocks["Name"].value' 里解出形态键名(沿用旧版对内嵌引号安全的切法)。"""
    interior = '['.join(data_path.split('[')[1:])
    interior = ']'.join(interior.split(']')[:-1])
    if len(interior) < 2:
        return None
    return codecs.decode(interior[1:-1], 'unicode_escape')


def _copy_fcurve_settings(source_fcurve, target_fcurve):
    """复制 F-Curve 的修改器、外插与关键帧。"""
    for modifier in list(target_fcurve.modifiers):
        target_fcurve.modifiers.remove(modifier)
    for modifier in source_fcurve.modifiers:
        copy = target_fcurve.modifiers.new(modifier.type)
        for property_definition in modifier.bl_rna.properties:
            if property_definition.is_readonly:
                continue
            identifier = property_definition.identifier
            try:
                setattr(copy, identifier, getattr(modifier, identifier))
            except (AttributeError, TypeError):
                pass
    target_fcurve.extrapolation = source_fcurve.extrapolation
    source_points = source_fcurve.keyframe_points
    if len(source_points) == 0:
        return
    target_points = target_fcurve.keyframe_points
    target_points.add(len(source_points))
    for source_point, target_point in zip(source_points, target_points):
        target_point.co = source_point.co
        target_point.interpolation = source_point.interpolation
        target_point.handle_left = source_point.handle_left
        target_point.handle_left_type = source_point.handle_left_type
        target_point.handle_right = source_point.handle_right
        target_point.handle_right_type = source_point.handle_right_type


_VARIABLE_TARGET_ATTRIBUTES = (
    "data_path", "bone_target", "transform_type", "transform_space",
    "rotation_mode", "context_property", "fallback_value", "use_fallback_value",
)


def _copy_variable_target(source_target, destination_target, identifier_remap):
    """复制单个驱动器变量目标,ID 引用按 remap 表替换(形态键数据块 / 骨架对象)。"""
    if getattr(source_target, "id_type", None) is not None:
        try:
            destination_target.id_type = source_target.id_type
        except AttributeError:
            pass  # 非 SINGLE_PROP 变量的 id_type 只读(锁定为 OBJECT)
    identifier = source_target.id
    identifier = identifier_remap.get(identifier, identifier)
    try:
        destination_target.id = identifier
    except (AttributeError, TypeError):
        pass
    for attribute in _VARIABLE_TARGET_ATTRIBUTES:
        if not hasattr(source_target, attribute):
            continue
        try:
            setattr(destination_target, attribute, getattr(source_target, attribute))
        except (AttributeError, TypeError):
            pass


def transfer_shape_key_drivers(source_object, target_object,
                               armature_source=None, armature_target=None):
    """把源对象形态键上的全部驱动器复制到目标对象的同名形态键。

    返回复制的驱动器数量;源/目标缺形态键或源无驱动器时返回 0。
    armature_source → armature_target 用于把变量里引用的骨架整体重映射。
    """
    source_shape_keys = source_object.data.shape_keys
    target_shape_keys = target_object.data.shape_keys
    if source_shape_keys is None or target_shape_keys is None:
        return 0
    animation_data = source_shape_keys.animation_data
    if animation_data is None or not animation_data.drivers:
        return 0

    identifier_remap = {source_shape_keys: target_shape_keys}
    if armature_source is not None and armature_target is not None:
        identifier_remap[armature_source] = armature_target

    target_key_blocks = target_shape_keys.key_blocks
    transferred_count = 0
    for source_fcurve in animation_data.drivers:
        shape_key_name = _extract_shape_key_name(source_fcurve.data_path)
        if shape_key_name is None or shape_key_name not in target_key_blocks:
            continue
        channel = source_fcurve.data_path.rsplit(".", 1)[-1]
        target_key_block = target_key_blocks[shape_key_name]
        try:
            target_key_block.driver_remove(channel)
        except TypeError:
            pass
        try:
            target_fcurve = target_key_block.driver_add(channel)
        except TypeError:
            continue
        _copy_fcurve_settings(source_fcurve, target_fcurve)

        source_driver = source_fcurve.driver
        target_driver = target_fcurve.driver
        target_driver.type = source_driver.type
        if hasattr(source_driver, "use_self"):
            target_driver.use_self = source_driver.use_self
        for source_variable in source_driver.variables:
            target_variable = target_driver.variables.new()
            target_variable.name = source_variable.name
            target_variable.type = source_variable.type
            for index, source_target in enumerate(source_variable.targets):
                _copy_variable_target(
                    source_target, target_variable.targets[index], identifier_remap)
        # 表达式最后赋值,确保变量齐备后一次性求值成功。
        target_driver.expression = source_driver.expression
        transferred_count += 1
    return transferred_count

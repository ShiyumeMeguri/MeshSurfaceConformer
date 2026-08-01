# Mesh Surface Conformer —— 通用表面贴合与数据互转。
# 自 mesh-data-transfer-2 全量重构:统一"聚合对应"内核(BVH/KD + 重心/线性插值),
# 两种用法共用同一套内核:
#   · 传输(Transfer):形状 / 形态键 / 顶点组 / UV / 颜色属性 / 自定义法线 整类搬运;
#   · 转换(Convert):任意通道 → 任意通道,UV 变顶点、顶点变 UV、法线变颜色、
#     权重变颜色……跨域(顶点 ⇄ 角点)与跨分量都自动折算,顶点结果默认落形态键。
# 映射方式与 Blender DataTransfer 修改器逐字对齐(顶点域 7 种 + 角点域 6 种),
# 另加本插件独有的 UV Interpolated 匹配;活动物体 = 被修改的目标,另一个选中的网格 = 源,
# 转换模式下只选一个物体 = 就地按序号转换。
# 关键修复:
#   · 任意拓扑差异下按最近表面插值匹配,不再强制最近顶点吸附(吸附是显式映射/选项);
#   · UV 匹配对合并/接缝顶点逐 loop 采样后收敛,同一顶点必然得到唯一位置;
#   · 角点域数据(UV/颜色/法线)逐面采样,永不跨 UV 岛渗色;
#   · 纯数据级实现,不再依赖 DataTransfer 修改器/改源 seam/模式切换。

bl_info = {
    "name": "Mesh Surface Conformer",
    "author": "ShiyumeMeguri",
    "description": "Conform shape, shape keys, vertex groups, UVs, colors and custom "
                   "normals onto another surface, and convert any mesh data channel "
                   "into any other one (UVs to vertices, normals to colors, ...)",
    "blender": (4, 2, 0),
    "version": (1, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Mesh Surface Conformer",
    "category": "Mesh",
}

if "bpy" in locals():
    import importlib
    importlib.reload(correspondence)
    importlib.reload(mesh_buffers)
    importlib.reload(channels)
    importlib.reload(shape_key_drivers)
    importlib.reload(properties)
    importlib.reload(conform_session)
    importlib.reload(operators)
    importlib.reload(user_interface)
else:
    from . import correspondence
    from . import mesh_buffers
    from . import channels
    from . import shape_key_drivers
    from . import properties
    from . import conform_session
    from . import operators
    from . import user_interface

import bpy
from bpy.props import PointerProperty

classes = properties.classes + operators.classes + user_interface.classes


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_surface_conformer = PointerProperty(
        type=properties.MeshSurfaceConformerSettings)
    make_links_menu = getattr(bpy.types, "VIEW3D_MT_make_links", None)
    if make_links_menu is not None:
        make_links_menu.append(user_interface.draw_make_links_menu)


def unregister():
    make_links_menu = getattr(bpy.types, "VIEW3D_MT_make_links", None)
    if make_links_menu is not None:
        make_links_menu.remove(user_interface.draw_make_links_menu)
    del bpy.types.Scene.mesh_surface_conformer
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

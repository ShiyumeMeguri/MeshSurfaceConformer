# Mesh Surface Conformer —— 以任意数据为基准的表面贴合与数据传输。
# 统一"聚合对应"内核(BVH/KD + 重心/线性插值)干两件正交的事:
#   · 匹配(Match By):拿任意一份数据当"两个网格的共同坐标系"找对应关系
#     —— 形状 / UV / 颜色 / 权重 / 属性 / 形态键 / 法线 / 序号;
#   · 传输(Transfer):形状 / 形态键 / 顶点组 / UV / 颜色属性 / 自定义法线 整类搬运。
# 另有 UV 镜像修复:UV 完全镜像的模型,选中顶点可以从对面那一半把坐标修回来。
# 活动物体 = 被修改的目标,另一个选中的网格 = 源。
# 关键特性:
#   · 任意拓扑差异下按最近表面插值匹配,不强制最近顶点吸附(吸附是显式选项);
#   · UV 匹配对合并/接缝顶点逐 loop 采样后收敛,同一顶点必然得到唯一位置;
#   · 角点域数据(UV/颜色/法线)逐面采样,永不跨 UV 岛渗色;
#   · 纯数据级实现,不依赖 DataTransfer 修改器/改源 seam/模式切换。

bl_info = {
    "name": "Mesh Surface Conformer",
    "author": "ShiyumeMeguri",
    "description": "Conform shape, shape keys, vertex groups, UVs, colors and custom "
                   "normals onto another surface, matched by any mesh data you like — "
                   "plus UV-mirror repair of vertex positions",
    "blender": (4, 2, 0),
    "version": (2, 0, 0),
    "location": "3D Viewport > Sidebar (N) > Mesh Surface Conformer",
    "category": "Mesh",
}

if "bpy" in locals():
    import importlib
    importlib.reload(correspondence)
    importlib.reload(mesh_buffers)
    importlib.reload(channels)
    importlib.reload(properties)
    importlib.reload(conform_session)
    importlib.reload(operators)
    importlib.reload(user_interface)
else:
    from . import correspondence
    from . import mesh_buffers
    from . import channels
    from . import properties
    from . import conform_session
    from . import operators
    from . import user_interface

import bpy
from bpy.props import PointerProperty

classes = properties.classes + operators.classes + user_interface.classes

# (菜单类型名, 追加的绘制函数)—— 注册与反注册共用一张表,不会漏掉任何一个。
_MENU_ENTRIES = (
    ("VIEW3D_MT_make_links", user_interface.draw_make_links_menu),
    ("VIEW3D_MT_edit_mesh_vertices", user_interface.draw_edit_mesh_vertex_menu),
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_surface_conformer = PointerProperty(
        type=properties.MeshSurfaceConformerSettings)
    for menu_name, draw_function in _MENU_ENTRIES:
        menu = getattr(bpy.types, menu_name, None)
        if menu is not None:
            menu.append(draw_function)


def unregister():
    for menu_name, draw_function in _MENU_ENTRIES:
        menu = getattr(bpy.types, menu_name, None)
        if menu is not None:
            menu.remove(draw_function)
    del bpy.types.Scene.mesh_surface_conformer
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

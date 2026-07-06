# Mesh Surface Conformer —— 通用表面贴合与数据传输。
# 自 mesh-data-transfer-2 全量重构:统一"插值最近表面"对应内核(BVH + 重心插值),
# 覆盖 形状 / 形态键 / 顶点组 / UV / 颜色属性 / 自定义法线 六类数据,
# 支持 Surface(3D) / UV / Topology 三种匹配域与 World / Local 两种空间。
# 关键修复:
#   · 任意拓扑差异下按最近表面插值匹配,不再最近顶点吸附(吸附降级为可选项);
#   · UV 匹配对合并/接缝顶点逐 loop 采样后收敛,同一顶点必然得到唯一位置;
#   · 角点域数据(UV/颜色/法线)逐面采样,永不跨 UV 岛渗色;
#   · 纯数据级实现,不再依赖 DataTransfer 修改器/改源 seam/模式切换。

bl_info = {
    "name": "Mesh Surface Conformer",
    "author": "ShiyumeMeguri",
    "description": "Conform shape, shape keys, vertex groups, UVs, colors and custom "
                   "normals onto another surface with interpolated closest-surface "
                   "matching across arbitrary topologies",
    "blender": (4, 2, 0),
    "version": (1, 0, 0),
    "location": "Properties > Object Data > Mesh Surface Conformer",
    "category": "Mesh",
}

if "bpy" in locals():
    import importlib
    importlib.reload(correspondence)
    importlib.reload(mesh_buffers)
    importlib.reload(shape_key_drivers)
    importlib.reload(conform_session)
    importlib.reload(properties)
    importlib.reload(operators)
    importlib.reload(user_interface)
else:
    from . import correspondence
    from . import mesh_buffers
    from . import shape_key_drivers
    from . import conform_session
    from . import properties
    from . import operators
    from . import user_interface

import bpy
from bpy.props import PointerProperty

classes = properties.classes + operators.classes + user_interface.classes


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.mesh_surface_conformer = PointerProperty(
        type=properties.MeshSurfaceConformerSettings)
    make_links_menu = getattr(bpy.types, "VIEW3D_MT_make_links", None)
    if make_links_menu is not None:
        make_links_menu.append(user_interface.draw_make_links_menu)


def unregister():
    make_links_menu = getattr(bpy.types, "VIEW3D_MT_make_links", None)
    if make_links_menu is not None:
        make_links_menu.remove(user_interface.draw_make_links_menu)
    del bpy.types.Object.mesh_surface_conformer
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

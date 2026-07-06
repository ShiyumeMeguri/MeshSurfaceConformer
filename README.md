# Mesh Surface Conformer

作者:ShiyumeMeguri · Blender 4.2+ / 5.x

通用表面贴合与数据传输插件(自 mesh-data-transfer-2 全量重构)。
面板位置:`Properties > Object Data > Mesh Surface Conformer`,
菜单入口:`Object > Link/Transfer Data > Conform Surface Data`。

## 能力

| 数据 | 域 | 说明 |
| --- | --- | --- |
| Shape | 顶点 | 顶点位置贴合到源表面,可写成形态键 |
| Shape Keys | 顶点 | 全部形态键按增量重放到目标 Basis,可连带驱动器 |
| Vertex Groups | 顶点 | 全组一次性插值传输,保组序/锁定标记 |
| UVs | 角点 | 逐面角采样,接缝安全,永不跨 UV 岛渗色 |
| Color Attributes | 顶点/角点 | 两种域自动识别,BYTE/FLOAT 类型保持 |
| Custom Normals | 角点 | 源最终拆分法线 → 目标自定义法线(世界模式自动旋转) |

匹配域:**Surface(3D 插值最近表面 / 法线投射)· UV(经 UV 空间)· Topology(序号直通)**;
空间:World / Local;统一影响管线:Mix × 顶点组遮罩 × 编辑选择 × 最大距离(带衰减)。

## 对旧版缺陷的修复

1. **插值最近表面匹配**:任意拓扑差异(低模↔低模/高模)都取源面上的精确最近点做重心插值,
   不再最近顶点吸附导致的顶点重叠/错位;"Snap to Vertices" 保留为显式可选项。
2. **UV 匹配的合并顶点一致性**:同一网格顶点的多个 UV loop(接缝/合并顶点)逐 loop 采样后
   按逆距离权重收敛,顶点必然落到唯一位置,修复旧版"最后写入者赢"的错乱。
3. **角点域接缝正确性**:UV/颜色/法线按"导向偏置"逐面角采样,接缝两侧各自命中正确源面,
   边界权重线性外推不内缩。
4. 纯数据级实现:去掉 DataTransfer 修改器、`seams_from_islands` 改源网格、模式切换链;
   求值源(Use Modified Source)时全部数组取自同一份求值网格,消除索引错位。
5. 现代 API:numpy 2.x 兼容(旧版 `np.float` 已崩)、`uv_layers[].uv`、`corner_normals`、
   `color_attributes`。

## 性能

- 一次会话只构建一次顶点域/角点域对应关系,六类数据复用同一内核;
- 全部读写走 `foreach_get/foreach_set` 精确 dtype 快路径,数学计算 float64 向量化;
- UV 查询按 2^24 量化去重(典型省 4~6 倍);顶点组写回按 16 位量化批量 `add`。

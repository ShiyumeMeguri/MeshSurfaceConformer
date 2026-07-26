# Mesh Surface Conformer

作者:ShiyumeMeguri · Blender 4.2+ / 5.x

通用表面贴合与数据互转插件(自 mesh-data-transfer-2 全量重构)。
面板位置:`3D 视图 > 侧边栏 (N) > Mesh Surface Conformer`(独立标签页),
菜单入口:`Object > Link/Transfer Data > Conform Surface Data`(Ctrl+L)。

## 两个模式,一个按钮

面板顶部只有两个标签页,选谁决定按钮做什么:

| 模式 | 干什么 | 怎么选物体 |
| --- | --- | --- |
| **Transfer** | 整类数据搬运:形状 / 形态键 / 顶点组 / UV / 颜色 / 法线 | 先选目标,最后选源(活动物体),支持一次多目标 |
| **Convert** | **任意通道 → 任意通道**,跨类型互转 | 只选一个 = 就地转换;选两个 = 从源转到目标 |

两个模式都受同一个 **Match By(匹配基准)** 支配 —— 用哪份数据当基准去找对应关系,
与要搬哪份数据完全正交。这是整个插件的骨架:**以任意数据为基准,转换任意数据**。

其余全部折进四个默认折叠的子面板(Data / Convert Options、Influence、
Advanced Mapping、Rigging Helpers),平时看不见。

### Transfer:三步

1. 先选目标,最后选源(与 `object.data_transfer` / Ctrl+L 完全相同的约定);
2. **Match By** 选基准(见下表),需要指定层的基准会跟出两个层选择框;
3. 点 **Shape / Weights / Detail / All** 快捷预设(或自己勾六个开关)→ **Transfer**。

形状默认写进形态键(`As Shape Key`),原始形状随时可退回。

### Convert:两行

**From** 一行、**To** 一行,中间一个 ⟳ 交换按钮,下面实时显示分量怎么排:

```
From: UV Map    [UVMap]
              ⟳
To:   Shape Key [UVMap]
2 → 3 components:   X Y 0
```

七种通道,任意一对都能转(共 49 种组合,全部有自测覆盖):

| 通道 | 域 | 分量 |
| --- | --- | --- |
| Vertex Position | 顶点 | 3 |
| Shape Key | 顶点 | 3 |
| UV Map | 角点 | 2 |
| Normal | 角点 | 3 |
| Color Attribute | 顶点/角点 | 4 |
| Vertex Group | 顶点 | 1 |
| Attribute(任意点/角点属性) | 顶点/角点 | 1~4 |

**顶点类结果默认落到形态键**(To 的默认项就是 Shape Key),网格本体不动。

常用组合:

- `UV → Shape Key` = 物理展开:顶点摊平到 UV 布局(即"利用 UV 转空间顶点");
- `Vertex Position → UV` = 把世界/局部坐标烘进 UV 层(即"利用空间转换 UV 顶点");
- `Normal → Color Attribute` + Remap `-1..1 to 0..1` = 法线烘成顶点色;
- `Vertex Group → Color Attribute` = 权重可视化(灰阶,Alpha 补 1);
- `Color Attribute → Vertex Group` = 顶点色转权重(取 RGB 平均,忽略 Alpha);
- `Vertex Position → Attribute` = 存一份坐标备份,之后再 `Attribute → Shape Key` 还原。

跨域自动折算:角点 → 顶点取同顶点各角点均值(接缝顶点会提示被平均了多少个),
顶点 → 角点直接展开,无损。

分量不匹配也自动处理(Convert Options 里可以改成逐分量手动指定 X/Y/Z/W/长度/平均/0/1):

- 分量数相同 → 原样;
- 目标只要 1 个 → 取前三个分量平均(忽略 Alpha);
- 源只有 1 个 → 广播(Alpha 补 1);
- 目标更少 → 取前若干个(位置→UV = XY);
- 目标更多 → 补 0,4 分量目标末位补 1(UV→位置 = XY0)。

数值重映射:None / Normalize 0..1 / -1..1↔0..1 / Scale + Offset,
在分量重排**之前**作用,所以补出来的常量 Alpha 不会被一起缩放。

## 匹配基准(Match By)—— 拿哪份数据当"两个网格的共同坐标系"

这是本插件的核心:**任何一个通道都能当基准**。两边读同一个通道,值相同的地方就是同一个
位置,所以量纲天然对齐,不需要任何归一化。

| Match By | 含义 | 典型场景 |
| --- | --- | --- |
| **Shape**(默认) | 3D 形状 | 常规贴合:衣服贴身体、低模贴高模 |
| **UV Map** | UV 坐标 | **两个 mesh 的 UV 完全一致、形状完全不一致 → 形状被拉回完全一致** |
| **Color Attribute** | 颜色值 | 拿烘好的 ID / 位置图当匹配空间 |
| **Vertex Group** | 单个权重值 | 一维基准,退化为"权重值最接近" |
| **Attribute** | 任意点/角点属性 | 自己烘一份匹配空间(最强的兜底) |
| **Shape Key** | 某个形态键的形状 | 按静止态匹配而不是按当前形状 |
| **Normal** | 法线方向 | 按朝向匹配 |
| **Index** | 顶点/角点序号 | 同拓扑,精确且瞬时 |
| **Custom** | 手选 Blender 原生映射 | 见下 |

反过来同理:**两个 mesh 形状完全一致、UV 完全不一致** → Match By 选 `Shape`、传输 `UVs`,
UV 就被拉回完全一致。这两个方向都有逐值比对的自测(误差 < 1e-4)。

**Method** 决定在基准空间里怎么取值:

- `Interpolated`(默认)—— 取源元素之间的精确插值点;两边基准值一致时结果是精确的;
- `Nearest` —— 只取最近的那一个源元素,数值一个不改地搬过来(离散数据/ID 用这个);
- `Projected` —— 沿目标法线投射(仅 Shape 基准有意义,其余会自动退回插值并提示)。

`Custom` 下可用的全集与 Blender DataTransfer 修改器逐字对齐 ——
顶点域:Topology · Nearest Vertex · Nearest Edge Vertex · Nearest Edge Interpolated ·
Nearest Face Vertex · Nearest Face Interpolated · Projected Face Interpolated · **UV Interpolated**;
角点域:Topology · Nearest Corner and Best Matching Normal ·
Nearest Corner and Best Matching Face Normal · Nearest Corner of Nearest Face ·
Nearest Face Interpolated · Projected Face Interpolated · **UV Interpolated**。

基准是角点域(UV/法线/角点色)时,同一顶点的多份值(接缝)按逆距离权重收敛到唯一结果;
基准是顶点域(形状/权重/点色)时,角点查询带"导向偏置",接缝两侧各归各的面。
就地转换(只选一个物体)恒为逐序号对应,精确且不做任何空间查询。

### 基准对不上时不会闷声出错

"目标被吸成一个点/一小团"只有一个成因:**两边的基准其实不是同一份数据**
(最常见是源用 `UVMap`、目标却用了另一套光照 UV,或那层根本没展开)。三道闸:

1. 面板直接写出**实际解析到的层**:`Matching 'UVMap' → 'UVMap'`,两边不同名时是红叹号;
2. 任何一侧的基准整层没有尺寸(全零 UV / 没展开)→ **当场报错**,一个顶点都不动;
3. 结果尺寸缩到目标原尺寸的 2% 以下 → 明确警告"两边基准可能对不上"。

目标层名的解析顺序是 **显式指定 > 与源同名 > 目标的活动层**,所以只要两边层名一致,
哪边的活动层是什么都不影响结果。

## 骨骼:按名字成批转换

Convert 模式勾上 **All, Matched by Name**,源上该类型的每一层/组/键都会转到目标上
**同名**的那一份 —— 整套骨骼权重一次搬完,不用一根根点:

- `Vertex Group → Vertex Group` + All = 全套骨骼权重按骨名配对转移(几何用上面任意基准匹配);
- `Vertex Group → Attribute` + All = 每根骨的权重变成同名属性;
- 形态键、UV 层、颜色属性同样支持成批按名配对。

Transfer 模式的 Vertex Groups / Shape Keys 本来就是按名字配对的整类搬运,
两条路都不依赖顶点组的序号。

空间:World / Local;统一影响管线:Mix × 顶点组遮罩 × 编辑选择 × 最大距离(带衰减)。

> 为什么不是修改器:Python 插件无法注册原生修改器;Geometry Nodes 写不了形态键/
> 自定义法线/批量顶点组,做成 GN 修改器只能交残血版,故按一次性算子交付(带撤销)。

## 对旧版缺陷的修复

1. **插值最近表面匹配**:任意拓扑差异(低模↔低模/高模)都取源面上的精确最近点做重心插值,
   不再强制最近顶点吸附(Nearest 系映射与 Snap 选项按 Blender 语义显式提供)。
2. **UV 匹配的合并顶点一致性**:同一网格顶点的多个 UV loop(接缝/合并顶点)逐 loop 采样后
   按逆距离权重收敛,顶点必然落到唯一位置,修复旧版"最后写入者赢"的错乱。
3. **角点域接缝正确性**:UV/颜色/法线按"导向偏置"逐面角采样,接缝两侧各自命中正确源面,
   边界权重线性外推不内缩。
4. 纯数据级实现:去掉 DataTransfer 修改器、`seams_from_islands` 改源网格、模式切换链;
   求值源(Use Modified Source)时全部数组取自同一份求值网格,消除索引错位。
5. 现代 API:numpy 2.x 兼容(旧版 `np.float` 已崩)、`uv_layers[].uv`、`corner_normals`、
   `color_attributes`。

## 性能

- 一次会话只构建一次顶点域/角点域对应关系,全部数据复用同一采样内核;
- 每种对应关系都实现统一的 `sample(data, domain)`,跨域折算只是一次 `bincount` /
  一次 gather,所以"任意通道 → 任意通道"没有额外查询成本;
- 全部读写走 `foreach_get/foreach_set` 精确 dtype 快路径,数学计算 float64 向量化;
- Nearest 系竞选(最近面顶点/最匹配法线角点)用 ragged 展开 + lexsort 分段取优,全程无 Python 逐候选循环;
- 边映射以退化三角形 (a, b, b) 建 BVH,`find_nearest` 天然退化为线段最近点;
- UV 查询按 2^24 量化去重(典型省 4~6 倍);顶点组写回按 16 位量化批量 `add`。

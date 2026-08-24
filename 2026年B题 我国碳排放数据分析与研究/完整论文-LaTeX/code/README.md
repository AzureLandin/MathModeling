# 代码说明（支撑材料 · 按问题拆分）

本目录按数模论文四问拆分为 **4 个互相独立的可运行脚本**，对应论文的「问题一～问题四」。
每个脚本都自带精简的 `load_inputs()` / `annualize_2025()` / `build_annual()`，**从原始附件重读、重算**，
互不依赖对方的 CSV 产物，因此**可以任意顺序单独运行**，也方便评委逐问对照。

| 文件 | 对应问题 | 核心方法 | 主要产物（写入 `results/`） |
|---|---|---|---|
| `problem_1.py` | 问题一：空间格局与聚类 | Moran's I 置换检验、Kruskal–Wallis、Ward 层次聚类 + 轮廓系数选 k | `q1_省级指标与分类.csv`、`q1_Moran检验.csv`、`q1_分组差异检验.csv`、`q1_PCA.csv`、`q1_聚类稳定性.csv`、`q1_聚类决策.csv` |
| `problem_2.py` | 问题二：驱动因素 | 带符号约束 STIRPAT 岭回归（人口/GDP/煤炭≥0、清洁≤0）+ 留一法选 λ + 扩展窗口回测 | `q2_驱动因素系数.csv`、`q2_回测.csv` |
| `problem_3.py` | 问题三：情景预测 | 复用 problem_2 模型，逐年递推能源结构，三情景（基准/低碳/强化低碳）2026–2045 预测 | `q3_2026_2045_三情景预测.csv`、`q3_情景约束检查.csv` |
| `problem_4.py` | 问题四：政策映射 | 复用 problem_1 分类 + problem_2 驱动 + problem_3 预测，做政策工具映射、部门排序、阶段指标 | `q4_省级政策映射.csv`、`q4_类别政策覆盖率.csv`、`q4_阶段情景指标.csv`、`q4_部门优先级.csv` |

> 代码层复用（仅 import 函数，CSV 结果彼此独立）：
> problem_3 `from problem_2 import drivers_and_model`；
> problem_4 `from problem_1 import classify` / `from problem_2 import drivers_and_model` / `from problem_3 import forecast_scenarios`。
> 这只是 **import 复用函数**，四问的 **CSV 结果彼此独立**，互不影响。

---

## 1. 依赖

- Python ≥ 3.10
- `numpy`、`pandas`、`scikit-learn`、`openpyxl`（读 `.xlsx` 附件2）

```bash
pip install numpy pandas scikit-learn openpyxl
```

## 2. 数据自包含（无需外部路径）

原始附件已复制到本工程 `data/` 目录，脚本默认从此读取，**不再依赖任何他人机器路径**：

```
完整论文-LaTeX/
├── code/            # 本目录：problem_1–4 脚本 + README
├── data/
│   ├── 附件1-中国2019年-2025年碳排放数据.csv
│   ├── 附件2-2022年30个省份排放清单.xlsx
│   └── 全国年度驱动变量_来源数据.csv
└── results/         # 运行后自动生成（四问各自的 CSV 产物）
```

## 3. 运行方式

每个脚本都支持命令行参数（均有默认值，最常用时**直接运行即可**）：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--project-root` | 工程根目录（`code/` 的上一级） | 脚本所在目录的上一级（自动推断） |
| `--input-root` | 原始附件所在目录 | `<project-root>/data` |
| `--seed` | 随机种子（聚类稳定性等） | `20260822` |

### 方式 A：在工程根目录下，每条命令跑一问（推荐，最直观）

```bash
cd "完整论文-LaTeX"
python code/problem_1.py
python code/problem_2.py
python code/problem_3.py
python code/problem_4.py
```

### 方式 B：显式指定工程根（任意目录均可启动）

```bash
python code/problem_1.py --project-root "E:/mathmodeling/MathModeling/2026年B题 我国碳排放数据分析与研究/完整论文-LaTeX"
python code/problem_2.py --project-root "E:/mathmodeling/MathModeling/2026年B题 我国碳排放数据分析与研究/完整论文-LaTeX"
python code/problem_3.py --project-root "E:/mathmodeling/MathModeling/2026年B题 我国碳排放数据分析与研究/完整论文-LaTeX"
python code/problem_4.py   --project-root "E:/mathmodeling/MathModeling/2026年B题 我国碳排放数据分析与研究/完整论文-LaTeX"
```

> 因四问完全独立，**任意顺序、单独运行均可**；想只复现某一问的结论，单独跑对应脚本即可。

## 4. 运行结果

- 每问运行结束会在终端打印一段 **JSON 摘要**（关键指标 / 分类数 / 预测区间等），便于快速核对。
- 详细数据写入 `results/`（见上表），可直接喂给论文表格与正文引用。
- problem_1 会额外产出清洗后的 `data/附件1_清洗后.csv`，供查阅，但**不是其他脚本的输入依赖**。

## 5. 归档说明

原 `All_*.py`（统一工具、绘图、文档构建、总运行器等 8 个文件）已移至 `code/archive/`，
仅作历史留档，**当前支撑材料不需要它们即可完整复现四问结果**。本工程已不含任何绘图代码（按需求精简）。

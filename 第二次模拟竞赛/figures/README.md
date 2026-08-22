# figures/ 图表说明

> 生成代码：`code/step1_datapreprocess.py::_plot_all`  渲染：matplotlib 3.11.1 + seaborn 0.13.2  分辨率：300 dpi  字体：SimHei / Microsoft YaHei  配色：Set2 / Blues_r / YlOrRd / RdBu_r

| 文件 | 尺寸 | 内容 | 坐标/图例 |
|------|------|------|-----------|
| `problem1_emission_total_rank.png` | 12×6 in | 30省总量横向排序柱状（低→高） | x: Mt CO₂, y: 省份中文, 柱上标注数值 |
| `problem1_four_features_boxplot.png` | 14×4.5 in | 四特征箱线+抖动点（1×4） | 总量/强度/人均/工业占比，附中位/均值 |
| `problem1_four_features_hist.png` | 11×8 in | 四特征直方图（2×2，10 bins） | 红虚=均值，绿点=中位，y=省份数 |
| `problem1_correlation_heatmap.png` | 6.2×5.2 in | Pearson 4×4 热力图 | RdBu_r, -1~1, 两位小数，中文缩写 |
| `problem1_industrial_share_vs_intensity.png` | 8×6 in | 工业占比 vs 强度（气泡大小=总量，颜色=总量） | 30省标注，YlOrRd 色条 Mt |
| `problem1_percapita_vs_total.png` | 8×6 in | 人均 vs 总量散点 | 30省标注，Set2 单色 |

均含标题、轴标签、图例、网格，可直接用于论文。重新生成：`E:\Anaconda\envs\math_modeling\python.exe code/step1_datapreprocess.py`

"""绘制问题一六张分阶段独立流程图 (学术出版级)。
严格依据 D:\\建模\\MathModeling\\2026国赛\\C题\\reports\\流程图绘制注意事项与示意结点.md 规范：
1. 每个阶段单独一张图，图面紧凑雅致，留白适度均衡；
2. 一个节点只表达一个动作或结论；
3. 严格遵循蓝/橙/绿/紫/灰/红标准学术配色；
4. 同时输出高分辨率 PNG (300 DPI) 和矢量图 SVG。
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

# -------------------------------------------------------------------------
# 1. 路径设置
# -------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output"
IMG_DIRS = [CURRENT_DIR, OUTPUT_DIR]

for d in IMG_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 2. 字体与风格
# -------------------------------------------------------------------------
def setup_style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    cn_font = "Microsoft YaHei"
    for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if f in installed:
            cn_font = f
            break

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cn_font, "SimHei", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })

# -------------------------------------------------------------------------
# 3. 颜色语义常量
# -------------------------------------------------------------------------
C_BLUE_BG   = "#EBF5FB"  # 浅海蓝 (输入/购电)
C_BLUE_EDGE = "#2980B9"
C_BLUE_TXT  = "#154360"

C_AMBER_BG   = "#FEF9E7" # 浅琥珀 (电价/光伏/目标)
C_AMBER_EDGE = "#D4AC0D"
C_AMBER_TXT  = "#7D6608"

C_GREEN_BG   = "#E8F8F5" # 浅翡翠绿 (储能/有效/通过)
C_GREEN_EDGE = "#27AE60"
C_GREEN_TXT  = "#145A32"

C_PURP_BG   = "#F4ECF7"  # 浅紫 (模型/HiGHS求解)
C_PURP_EDGE = "#8E44AD"
C_PURP_TXT  = "#512E5F"

C_GRAY_BG   = "#F8F9F9"  # 浅灰 (约束/检查/常规步骤)
C_GRAY_EDGE = "#7F8C8D"
C_GRAY_TXT  = "#2C3E50"

C_RED_BG    = "#FDEDEC"  # 浅珊瑚红 (未通过/返回)
C_RED_EDGE  = "#E74C3C"
C_RED_TXT   = "#922B21"

# -------------------------------------------------------------------------
# 4. 通用绘图组件
# -------------------------------------------------------------------------
def draw_card(ax, x, y, w, h, title, subtext="", bg=C_GRAY_BG, edge=C_GRAY_EDGE,
              txt_color=None, lw=1.3, radius=0.15, ls="-", title_size=10.5, sub_size=8.5):
    """绘制圆角矩形卡片 (x, y 为中心坐标)"""
    tc = txt_color if txt_color else edge
    x0 = x - w / 2
    y0 = y - h / 2
    box = FancyBboxPatch((x0, y0), w, h,
                         boxstyle=f"round,pad=0.0,rounding_size={radius}",
                         facecolor=bg, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(box)
    
    if title and subtext:
        ax.text(x, y + h * 0.16, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=tc, zorder=3)
        ax.text(x, y - h * 0.20, subtext, ha="center", va="center",
                fontsize=sub_size, color="#4A5568", zorder=3)
    elif title:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=tc, zorder=3)

def draw_rect(ax, x, y, w, h, title, subtext="", bg=C_PURP_BG, edge=C_PURP_EDGE,
              txt_color=None, lw=1.4, ls="-", title_size=11.0, sub_size=8.5):
    """绘制直角矩形 (常用于模型/求解器内核)"""
    tc = txt_color if txt_color else edge
    x0 = x - w / 2
    y0 = y - h / 2
    box = FancyBboxPatch((x0, y0), w, h,
                         boxstyle="square,pad=0.0",
                         facecolor=bg, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(box)
    
    if title and subtext:
        ax.text(x, y + h * 0.16, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=tc, zorder=3)
        ax.text(x, y - h * 0.20, subtext, ha="center", va="center",
                fontsize=sub_size, color="#4A5568", zorder=3)
    elif title:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=tc, zorder=3)

def draw_parallelogram(ax, x, y, w, h, title, subtext="", bg=C_BLUE_BG, edge=C_BLUE_EDGE,
                       txt_color=None, lw=1.3, slant=0.12, title_size=10.5, sub_size=8.5):
    """绘制平行四边形 (常用于数据输入与输出)"""
    tc = txt_color if txt_color else edge
    x0 = x - w / 2
    y0 = y - h / 2
    dx = w * slant
    pts = [
        [x0 + dx, y0],
        [x0 + w, y0],
        [x0 + w - dx, y0 + h],
        [x0, y0 + h]
    ]
    poly = Polygon(pts, closed=True, facecolor=bg, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(poly)
    
    if title and subtext:
        ax.text(x, y + h * 0.16, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=tc, zorder=3)
        ax.text(x, y - h * 0.20, subtext, ha="center", va="center",
                fontsize=sub_size, color="#4A5568", zorder=3)
    elif title:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=tc, zorder=3)

def draw_diamond(ax, x, y, w, h, title, bg=C_AMBER_BG, edge=C_AMBER_EDGE, txt_color=None, lw=1.3, title_size=10.0):
    """绘制菱形判断节点"""
    tc = txt_color if txt_color else edge
    pts = [
        [x, y - h / 2],
        [x + w / 2, y],
        [x, y + h / 2],
        [x - w / 2, y]
    ]
    poly = Polygon(pts, closed=True, facecolor=bg, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(poly)
    ax.text(x, y, title, ha="center", va="center", fontsize=title_size, weight="bold", color=tc, zorder=3)

def draw_arrow(ax, p1, p2, color="#5D6D7E", lw=1.3, rad=0.0, ls="-", text="", text_pos=0.5, text_offset=(0, 0.15), fontsize=8.5):
    """绘制箭头连接线"""
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
            mutation_scale=11, connectionstyle=f"arc3,rad={rad}"
        ),
        zorder=1
    )
    if text:
        tx = p1[0] + (p2[0] - p1[0]) * text_pos + text_offset[0]
        ty = p1[1] + (p2[1] - p1[1]) * text_pos + text_offset[1]
        ax.text(tx, ty, text, ha="center", va="center", fontsize=fontsize, weight="bold", color=color, zorder=4)

def save_fig(fig, base_name):
    """同时输出高分辨率 PNG 与矢量图 SVG"""
    for d in IMG_DIRS:
        png_path = d / f"{base_name}.png"
        svg_path = d / f"{base_name}.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f" Saved: {base_name}.png / .svg")

# -------------------------------------------------------------------------
# 5. 图1：问题一模型输入组成
# -------------------------------------------------------------------------
def draw_fig1_inputs():
    fig, ax = plt.subplots(figsize=(10.0, 5.0), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # 标题与图注
    ax.text(5.0, 4.6, "图1 问题一模型输入组成", ha="center", va="center",
            fontsize=13.5, weight="bold", color="#1B2631")
    ax.text(5.0, 0.35, "注：附件1提供全天分时电价、小区负载与光伏预测；储能系统设定初末状态相等 (E(0) = E(144) = 6000 kWh)。",
            ha="center", va="center", fontsize=8.2, color="#566573")

    # 左侧：三路原始数据输入 (平行四边形)
    draw_parallelogram(ax, 1.6, 3.6, 2.0, 0.75, "分时电价", "0.25 ~ 1.05 元/kWh", bg=C_AMBER_BG, edge=C_AMBER_EDGE, txt_color=C_AMBER_TXT)
    draw_parallelogram(ax, 1.6, 2.4, 2.0, 0.75, "小区负载", "预测负荷功率 (kW)", bg=C_BLUE_BG, edge=C_BLUE_EDGE, txt_color=C_BLUE_TXT)
    draw_parallelogram(ax, 1.6, 1.2, 2.0, 0.75, "光伏预测", "预测出力功率 (kW)", bg=C_AMBER_BG, edge=C_AMBER_EDGE, txt_color=C_AMBER_TXT)

    # 中间上方：时序数据集
    draw_card(ax, 4.8, 2.4, 2.2, 1.1, "附件1时序数据", "144个时段 · 间隔10分钟", bg=C_BLUE_BG, edge=C_BLUE_EDGE, txt_color=C_BLUE_TXT, radius=0.12)

    # 中间下方：储能物理参数
    draw_card(ax, 4.8, 1.2, 2.2, 0.85, "储能物理参数", "容量/功率限值与效率", bg=C_GREEN_BG, edge=C_GREEN_EDGE, txt_color=C_GREEN_TXT, radius=0.12)

    # 右侧：模型参数集
    draw_card(ax, 8.2, 1.8, 2.2, 1.5, "模型输入参数集", "时序矩阵与物理边界\n统一对齐", bg="#E8EAF6", edge="#283593", txt_color="#1A237E", radius=0.15)

    # 箭头连接
    draw_arrow(ax, (2.6, 3.6), (3.7, 2.6))
    draw_arrow(ax, (2.6, 2.4), (3.7, 2.4))
    draw_arrow(ax, (2.6, 1.2), (3.7, 2.2))
    draw_arrow(ax, (5.9, 2.4), (7.1, 2.0))
    draw_arrow(ax, (5.9, 1.2), (7.1, 1.6))

    save_fig(fig, "图1_问题一模型输入组成")

# -------------------------------------------------------------------------
# 6. 图2：附件一数据预处理流程
# -------------------------------------------------------------------------
def draw_fig2_preprocessing():
    fig, ax = plt.subplots(figsize=(11.5, 3.6), dpi=300)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    # 标题与图注
    ax.text(5.75, 3.2, "图2 附件一数据预处理流程", ha="center", va="center",
            fontsize=13.5, weight="bold", color="#1B2631")
    ax.text(5.75, 0.35, "注：功率转电量公式为 E = P · Δt (kWh)；严格校验时段连续性与越界异常，输出标准化时序矩阵。",
            ha="center", va="center", fontsize=8.2, color="#566573")

    steps = [
        ("读取原始数据", "CSV/Excel解析", C_GRAY_BG, C_GRAY_EDGE, C_GRAY_TXT),
        ("统一时间轴", "00:00 ~ 24:00", C_GRAY_BG, C_GRAY_EDGE, C_GRAY_TXT),
        ("映射10分钟时段", "采样点代表前一时段", C_BLUE_BG, C_BLUE_EDGE, C_BLUE_TXT),
        ("功率转换为电量", "时间步长 Δt = 1/6 h", C_AMBER_BG, C_AMBER_EDGE, C_AMBER_TXT),
        ("检查数据完整性", "无缺失与数值越界", C_GRAY_BG, C_GRAY_EDGE, C_GRAY_TXT),
        ("生成标准输入表", "标准化矩阵交付模型", C_GREEN_BG, C_GREEN_EDGE, C_GREEN_TXT),
    ]

    xs = [1.0 + i * 1.88 for i in range(6)]
    y_node = 1.75
    w_node, h_node = 1.62, 1.05

    for i, (title, sub, bg, edge, txt_col) in enumerate(steps):
        draw_card(ax, xs[i], y_node, w_node, h_node, title, sub, bg=bg, edge=edge, txt_color=txt_col, radius=0.12)
        if i < 5:
            draw_arrow(ax, (xs[i] + w_node / 2, y_node), (xs[i+1] - w_node / 2, y_node))

    save_fig(fig, "图2_附件一数据预处理流程")

# -------------------------------------------------------------------------
# 7. 图3：确定性MILP模型结构
# -------------------------------------------------------------------------
def draw_fig3_model_structure():
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    # 标题与图注
    ax.text(5.25, 6.4, "图3 确定性MILP模型结构", ha="center", va="center",
            fontsize=13.5, weight="bold", color="#1B2631")
    ax.text(5.25, 0.35, "注：MILP以全天购电费用最小为目标，通过二进制变量严格约束充放互斥，满足供需平衡、状态递推与日循环条件。",
            ha="center", va="center", fontsize=8.2, color="#566573")

    # 顶部目标函数
    draw_card(ax, 5.25, 5.35, 4.2, 0.85, "目标函数：最小化全天购电费", "min ∑ c(t) · P_grid(t) · Δt",
              bg=C_AMBER_BG, edge=C_AMBER_EDGE, txt_color=C_AMBER_TXT, radius=0.12)

    # 中心核心模型 (直角矩形)
    draw_rect(ax, 5.25, 3.8, 3.6, 1.15, "确定性 MILP 核心模型", "混合整数线性规划",
              bg=C_PURP_BG, edge=C_PURP_EDGE, txt_color=C_PURP_TXT, lw=1.6)

    # 左侧决策变量
    draw_card(ax, 1.65, 3.8, 2.3, 1.15, "决策变量", "购电量 · 充电量\n放电量 · 互斥0-1变量",
              bg=C_BLUE_BG, edge=C_BLUE_EDGE, txt_color=C_BLUE_TXT, radius=0.12)

    # 右侧状态变量
    draw_card(ax, 8.85, 3.8, 2.3, 1.15, "状态变量", "各时段储电量 E(t)\nSOC 物理动态跟踪",
              bg=C_GREEN_BG, edge=C_GREEN_EDGE, txt_color=C_GREEN_TXT, radius=0.12)

    # 箭头汇聚到核心模型
    draw_arrow(ax, (5.25, 4.92), (5.25, 4.38))
    draw_arrow(ax, (2.80, 3.80), (3.45, 3.80))
    draw_arrow(ax, (7.70, 3.80), (7.05, 3.80))

    # 下方 6 个独立约束节点 (2行3列分布)
    constraints = [
        ("母线供需平衡", "负荷=光伏+购电+放电-充电"),
        ("储能状态递推", "E(t)=E(t-1)+η_c·E_c-E_d/η_d"),
        ("储能容量边界", "1200 ≤ E(t) ≤ 10800 kWh"),
        ("充放功率限制", "充放电功率 ≤ 5000 kW"),
        ("充放严格互斥", "u_c(t) + u_d(t) ≤ 1"),
        ("初末状态相等", "E(0) = E(144) = 6000 kWh"),
    ]

    c_xs = [2.2, 5.25, 8.3]
    c_ys = [2.2, 1.1]

    idx = 0
    for r in range(2):
        for c in range(3):
            title, sub = constraints[idx]
            draw_card(ax, c_xs[c], c_ys[r], 2.65, 0.78, title, sub,
                      bg=C_GRAY_BG, edge=C_GRAY_EDGE, txt_color=C_GRAY_TXT, radius=0.10)
            idx += 1

    # 约束集合整体汇入中心模型
    draw_arrow(ax, (2.2, 2.6), (4.2, 3.25))
    draw_arrow(ax, (5.25, 2.6), (5.25, 3.22))
    draw_arrow(ax, (8.3, 2.6), (6.3, 3.25))

    save_fig(fig, "图3_确定性MILP模型结构")

# -------------------------------------------------------------------------
# 8. 图4：MILP优化求解流程
# -------------------------------------------------------------------------
def draw_fig4_solving():
    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=300)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # 标题与图注
    ax.text(4.25, 6.8, "图4 MILP优化求解流程", ha="center", va="center",
            fontsize=13.5, weight="bold", color="#1B2631")
    ax.text(4.25, 0.35, "注：HiGHS 求解器结合内点法与分支定界算法，快速获得全天调度全局最优解。",
            ha="center", va="center", fontsize=8.2, color="#566573")

    # 步骤1：载入标准数据
    draw_card(ax, 4.25, 5.95, 3.4, 0.8, "载入标准数据", "输入电价/负荷/光伏及参数",
              bg=C_GRAY_BG, edge=C_GRAY_EDGE, txt_color=C_GRAY_TXT, radius=0.12)

    # 步骤2：构建MILP矩阵
    draw_card(ax, 4.25, 4.75, 3.4, 0.8, "构建MILP矩阵", "生成系数矩阵 A_ub, A_eq, b, c",
              bg=C_GRAY_BG, edge=C_GRAY_EDGE, txt_color=C_GRAY_TXT, radius=0.12)

    # 步骤3：并列分支 (边界与整型)
    draw_card(ax, 2.45, 3.55, 2.6, 0.8, "设置变量上下界", "决策与状态物理极值",
              bg=C_BLUE_BG, edge=C_BLUE_EDGE, txt_color=C_BLUE_TXT, radius=0.12)
    draw_card(ax, 6.05, 3.55, 2.6, 0.8, "指定整数变量", "充放状态 0-1 二进制变量",
              bg=C_AMBER_BG, edge=C_AMBER_EDGE, txt_color=C_AMBER_TXT, radius=0.12)

    # 步骤4：HiGHS 求解内核 (直角矩形)
    draw_rect(ax, 4.25, 2.25, 4.0, 0.95, "调用 HiGHS 求解器", "分支定界 (Branch & Cut) 求解",
              bg=C_PURP_BG, edge=C_PURP_EDGE, txt_color=C_PURP_TXT, lw=1.5)

    # 步骤5：输出候选解
    draw_card(ax, 4.25, 1.10, 3.4, 0.75, "获得候选最优解", "提取决策向量与各时段状态",
              bg=C_GREEN_BG, edge=C_GREEN_EDGE, txt_color=C_GREEN_TXT, radius=0.12)

    # 箭头连接
    draw_arrow(ax, (4.25, 5.55), (4.25, 5.15))
    draw_arrow(ax, (3.5, 4.35), (2.45, 3.95))
    draw_arrow(ax, (5.0, 4.35), (6.05, 3.95))
    draw_arrow(ax, (2.45, 3.15), (3.5, 2.73))
    draw_arrow(ax, (6.05, 3.15), (5.0, 2.73))
    draw_arrow(ax, (4.25, 1.77), (4.25, 1.48))

    save_fig(fig, "图4_MILP优化求解流程")

# -------------------------------------------------------------------------
# 9. 图5：调度结果可行性与最优性验证
# -------------------------------------------------------------------------
def draw_fig5_verification():
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    # 标题与图注
    ax.text(5.25, 6.4, "图5 调度结果可行性与最优性验证", ha="center", va="center",
            fontsize=13.5, weight="bold", color="#1B2631")
    ax.text(5.25, 0.35, "注：经严苛复核，物理约束最大残差 < 1e-12 kWh，充放互斥率 100%，MIP Gap = 0 确认全局最优解。",
            ha="center", va="center", fontsize=8.2, color="#566573")

    # 左侧输入
    draw_card(ax, 1.4, 3.6, 1.9, 0.9, "候选最优解", "提取求解器结果",
              bg=C_GRAY_BG, edge=C_GRAY_EDGE, txt_color=C_GRAY_TXT, radius=0.12)
    draw_card(ax, 3.7, 3.6, 1.9, 0.9, "执行多维核验", "分流并列检查",
              bg=C_BLUE_BG, edge=C_BLUE_EDGE, txt_color=C_BLUE_TXT, radius=0.12)

    draw_arrow(ax, (2.35, 3.6), (2.75, 3.6))

    # 中间 5 个独立校验节点
    checks = [
        ("供需平衡核验", "残差 < 1e-12"),
        ("状态递推核验", "残差 < 1e-12"),
        ("容量功率核验", "无越界违规"),
        ("充放互斥核验", "无同充同放"),
        ("初末状态核验", "E(0)=E(144)"),
    ]
    y_checks = [5.3, 4.45, 3.6, 2.75, 1.9]
    for i, (title, sub) in enumerate(checks):
        draw_card(ax, 6.1, y_checks[i], 2.1, 0.65, title, sub,
                  bg=C_GRAY_BG, edge=C_GRAY_EDGE, txt_color=C_GRAY_TXT, radius=0.10)
        draw_arrow(ax, (4.65, 3.6), (5.05, y_checks[i]))

    # 右侧菱形判断
    draw_diamond(ax, 8.7, 3.6, 1.7, 1.2, "是否全部\n通过?",
                 bg=C_AMBER_BG, edge=C_AMBER_EDGE, txt_color=C_AMBER_TXT, title_size=9.5)

    for i in range(5):
        draw_arrow(ax, (7.15, y_checks[i]), (7.85, 3.6))

    # 分支1：通过 (绿色)
    draw_card(ax, 8.7, 1.6, 2.3, 1.05, "确认调度有效解", "MIP Gap = 0\n最大残差 < 1e-12 kWh",
              bg=C_GREEN_BG, edge=C_GREEN_EDGE, txt_color=C_GREEN_TXT, radius=0.12)
    draw_arrow(ax, (8.7, 3.0), (8.7, 2.13), text="是", text_offset=(0.25, 0.0), color=C_GREEN_EDGE)

    # 分支2：未通过 (红色)
    draw_card(ax, 5.0, 6.0, 2.2, 0.65, "返回检查模型与参数", "修正时序/参数/约束",
              bg=C_RED_BG, edge=C_RED_EDGE, txt_color=C_RED_TXT, radius=0.10)
    draw_arrow(ax, (8.7, 4.2), (8.7, 6.0), color=C_RED_EDGE)
    draw_arrow(ax, (8.7, 6.0), (6.1, 6.0), text="否", text_offset=(0.0, 0.2), color=C_RED_EDGE)
    draw_arrow(ax, (3.9, 6.0), (3.7, 4.05), ls="--", color=C_RED_EDGE)

    save_fig(fig, "图5_调度结果可行性与最优性验证")

# -------------------------------------------------------------------------
# 10. 图6：问题一调度结果及基准对照
# -------------------------------------------------------------------------
def draw_fig6_results():
    fig, ax = plt.subplots(figsize=(11.0, 6.8), dpi=300)
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    # 标题与图注
    ax.text(5.5, 6.4, "图6 问题一调度结果及基准对照", ha="center", va="center",
            fontsize=13.5, weight="bold", color="#1B2631")
    ax.text(5.5, 0.35, "注：无储能购电费 48051.87 元，储能套利削峰填谷效果显著；LP松弛解验证费用一致性，MILP确保互斥有效。",
            ha="center", va="center", fontsize=8.2, color="#566573")

    # 左侧：输出调度方案明细
    draw_card(ax, 2.5, 5.3, 3.6, 0.75, "有效调度方案", "全局最优决策指令集",
              bg=C_BLUE_BG, edge=C_BLUE_EDGE, txt_color=C_BLUE_TXT, radius=0.12)

    outputs = [
        ("计划购电曲线", "144时段购电量"),
        ("储能充放计划", "充电/放电功率时序"),
        ("时序储能状态", "储电量与SOC演化"),
        ("分时电费明细", "各区间购电支出"),
    ]
    oxs = [1.5, 3.5, 1.5, 3.5]
    oys = [4.2, 4.2, 3.1, 3.1]
    for i in range(4):
        t, s = outputs[i]
        draw_card(ax, oxs[i], oys[i], 1.8, 0.72, t, s,
                  bg=C_GRAY_BG, edge=C_GRAY_EDGE, txt_color=C_GRAY_TXT, radius=0.10)
        draw_arrow(ax, (2.5, 4.92), (oxs[i], oys[i] + 0.36))

    # 右侧：方案基准对照
    draw_card(ax, 6.6, 5.1, 2.3, 0.75, "MILP 最优方案", "总费用 35126.95 元",
              bg=C_GREEN_BG, edge=C_GREEN_EDGE, txt_color=C_GREEN_TXT, radius=0.12)
    draw_card(ax, 6.6, 3.9, 2.3, 0.75, "无储能基准方案", "总费用 48051.87 元",
              bg=C_GRAY_BG, edge=C_GRAY_EDGE, txt_color=C_GRAY_TXT, radius=0.12)

    draw_card(ax, 9.6, 4.5, 2.0, 0.9, "综合效益对比", "套利降费 · 削峰填谷",
              bg=C_AMBER_BG, edge=C_AMBER_EDGE, txt_color=C_AMBER_TXT, radius=0.12)

    draw_arrow(ax, (7.75, 5.1), (8.6, 4.7))
    draw_arrow(ax, (7.75, 3.9), (8.6, 4.3))

    # LP 松弛对照 (虚线)
    draw_card(ax, 6.6, 2.7, 2.3, 0.65, "LP 连续松弛解", "费用 35126.95 元",
              bg=C_PURP_BG, edge=C_PURP_EDGE, txt_color=C_PURP_TXT, ls="--", radius=0.10)
    draw_card(ax, 9.6, 2.7, 2.0, 0.65, "最优费用完全一致", "数学下界验证成功",
              bg=C_PURP_BG, edge=C_PURP_EDGE, txt_color=C_PURP_TXT, ls="--", radius=0.10)
    draw_arrow(ax, (7.75, 2.7), (8.6, 2.7), ls="--", color=C_PURP_EDGE)

    # 底部 3 个数字指标卡
    draw_card(ax, 2.1, 1.25, 2.6, 0.95, "最优总购电费", "35,126.95 元",
              bg=C_AMBER_BG, edge=C_AMBER_EDGE, txt_color=C_AMBER_TXT, radius=0.12, title_size=10.0, sub_size=10.5)
    draw_card(ax, 5.5, 1.25, 2.6, 0.95, "全天净购电量", "59,482.70 kWh",
              bg=C_BLUE_BG, edge=C_BLUE_EDGE, txt_color=C_BLUE_TXT, radius=0.12, title_size=10.0, sub_size=10.5)
    draw_card(ax, 8.9, 1.25, 2.6, 0.95, "相对基准节省比例", "26.90 %",
              bg=C_GREEN_BG, edge=C_GREEN_EDGE, txt_color=C_GREEN_TXT, radius=0.12, title_size=10.0, sub_size=11.0)

    save_fig(fig, "图6_问题一调度结果及基准对照")

# -------------------------------------------------------------------------
# 11. 主程序
# -------------------------------------------------------------------------
def main():
    setup_style()
    print("开始批量绘制问题一六张分阶段独立流程图...")
    draw_fig1_inputs()
    draw_fig2_preprocessing()
    draw_fig3_model_structure()
    draw_fig4_solving()
    draw_fig5_verification()
    draw_fig6_results()
    print("全部 6 张流程图已成功绘制并保存！")

if __name__ == "__main__":
    main()

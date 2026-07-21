# -*- coding: utf-8 -*-
"""
可视化模块：生成论文所需8张图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay, ConvexHull
from utils import compute_mst

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FIGURES_DIR = r'E:\MathModeling\模块二练习\figures'


def plot_fig1(coords):
    """图1：供水点空间分布图"""
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(coords[1:, 0], coords[1:, 1], c='#3498DB', s=30, zorder=3, label='供水点 (P1-P180)')
    ax.scatter(coords[0, 0], coords[0, 1], c='#E74C3C', s=300, marker='*', zorder=5, label='中心供水站 A')
    ax.annotate('A (26, 31)', xy=(26, 31), xytext=(28, 28),
                fontsize=11, fontweight='bold', color='#E74C3C',
                arrowprops=dict(arrowstyle='->', color='#E74C3C'))
    ax.set_xlabel('X 坐标 (km)', fontsize=12)
    ax.set_ylabel('Y 坐标 (km)', fontsize=12)
    ax.set_title('图1  供水点空间分布', fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/图1_供水点空间分布.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/图1_供水点空间分布.pdf', bbox_inches='tight')
    plt.close()
    print("  图1 已保存")


def plot_fig2(coords, dist_matrix):
    """图2：Delaunay三角剖分与全局MST"""
    tri = Delaunay(coords)
    fig, ax = plt.subplots(figsize=(10, 9))

    edges_plotted = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                edge = (min(a, b), max(a, b))
                if edge not in edges_plotted:
                    edges_plotted.add(edge)
                    ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                            c='#BDC3C7', linewidth=0.5, zorder=1)

    # 全局MST
    _, mst_edges = compute_mst(set(range(181)), dist_matrix)
    for (u, v) in mst_edges:
        ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
                c='#E67E22', linewidth=1.5, alpha=0.8, zorder=2)

    ax.scatter(coords[1:, 0], coords[1:, 1], c='#3498DB', s=15, zorder=3)
    ax.scatter(coords[0, 0], coords[0, 1], c='#E74C3C', s=200, marker='*', zorder=5)

    legend_elements = [
        Line2D([0], [0], color='#BDC3C7', linewidth=1, label=f'Delaunay候选边 ({len(edges_plotted)}条)'),
        Line2D([0], [0], color='#E67E22', linewidth=2, label=f'全局MST ({len(mst_edges)}条)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    ax.set_title('图2  Delaunay三角剖分与全局MST', fontsize=14)
    ax.set_xlabel('X 坐标 (km)', fontsize=12)
    ax.set_ylabel('Y 坐标 (km)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/图2_Delaunay三角剖分.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/图2_Delaunay三角剖分.pdf', bbox_inches='tight')
    plt.close()
    print("  图2 已保存")


def plot_fig3(coords, clusters, primary_stations, cluster_mst_lengths, cluster_mst_edges):
    """图3：约束聚类结果图"""
    fig, ax = plt.subplots(figsize=(11, 10))
    K = len(clusters)
    cmap = cm.get_cmap('tab20', K)

    for k in range(K):
        cluster = clusters[k]
        nodes = list(cluster)
        color = cmap(k)

        ax.scatter(coords[nodes, 0], coords[nodes, 1], c=[color], s=40, zorder=3,
                   edgecolors='white', linewidths=0.5)

        for (u, v) in cluster_mst_edges[k]:
            ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
                    c=color, linewidth=1.0, alpha=0.7, zorder=2)

        if len(nodes) >= 3:
            try:
                hull = ConvexHull(coords[nodes])
                hull_pts = coords[nodes][hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], '--', c=color, linewidth=1.2, alpha=0.5, zorder=1)
            except Exception:
                pass

        ps = primary_stations[k]
        ax.scatter(coords[ps, 0], coords[ps, 1], c='black', s=120, marker='s', zorder=5,
                   edgecolors='white', linewidths=1.5)

        centroid = coords[nodes].mean(axis=0)
        ax.annotate(f'C{k+1}\n({cluster_mst_lengths[k]:.1f}km)',
                    xy=centroid, fontsize=8, ha='center', color=color, fontweight='bold')

    ax.scatter(coords[0, 0], coords[0, 1], c='#E74C3C', s=300, marker='*', zorder=6)
    ax.annotate('A', xy=(26, 31), xytext=(27, 29), fontsize=12, fontweight='bold', color='#E74C3C')

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='一级供水站'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#E74C3C', markersize=15, label='中心站 A'),
        Line2D([0], [0], linestyle='--', color='gray', linewidth=1.2, label='簇边界(凸包)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    ax.set_title('图3  约束凝聚聚类结果（各簇MST长度标注）', fontsize=14)
    ax.set_xlabel('X 坐标 (km)', fontsize=12)
    ax.set_ylabel('Y 坐标 (km)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/图3_聚类结果.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/图3_聚类结果.pdf', bbox_inches='tight')
    plt.close()
    print("  图3 已保存")


def plot_fig4(coords, E_I, primary_stations, L_I, dist_matrix):
    """图4：一级站选址与I型管网图"""
    fig, ax = plt.subplots(figsize=(10, 9))

    ax.scatter(coords[1:, 0], coords[1:, 1], c='#ECF0F1', s=15, zorder=1)

    for (u, v) in E_I:
        ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
                c='#E74C3C', linewidth=2.5, zorder=3, solid_capstyle='round')
        mid = (coords[u] + coords[v]) / 2
        length = dist_matrix[u][v]
        ax.annotate(f'{length:.1f}', xy=mid, fontsize=7, color='#C0392B',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))

    for ps in primary_stations:
        ax.scatter(coords[ps, 0], coords[ps, 1], c='#2C3E50', s=150, marker='s', zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.annotate(f'P{ps}', xy=(coords[ps, 0], coords[ps, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')

    ax.scatter(coords[0, 0], coords[0, 1], c='#E74C3C', s=350, marker='*', zorder=6)
    ax.annotate('A', xy=(26, 31), xytext=(27, 29), fontsize=13, fontweight='bold', color='#E74C3C')

    ax.set_title(f'图4  I型管网（一级站MST，总长 {L_I:.2f} km）', fontsize=14)
    ax.set_xlabel('X 坐标 (km)', fontsize=12)
    ax.set_ylabel('Y 坐标 (km)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/图4_I型管网.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/图4_I型管网.pdf', bbox_inches='tight')
    plt.close()
    print("  图4 已保存")


def plot_fig5(coords, E_I, E_II, primary_stations, L_I, L_II, C_total):
    """图5：完整管网拓扑图（最终方案）"""
    fig, ax = plt.subplots(figsize=(12, 11))

    # II型管道（蓝色细线）
    for edges in E_II:
        for (u, v) in edges:
            ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
                    c='#3498DB', linewidth=1.0, alpha=0.8, zorder=2)

    # I型管道（红色粗线）
    for (u, v) in E_I:
        ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
                c='#E74C3C', linewidth=2.5, zorder=3, solid_capstyle='round')

    # 二级站
    T_nodes = list(set(range(1, 181)) - set(primary_stations))
    ax.scatter(coords[T_nodes, 0], coords[T_nodes, 1], c='#3498DB', s=25, zorder=4,
               edgecolors='white', linewidths=0.3)

    # 一级站
    for ps in primary_stations:
        ax.scatter(coords[ps, 0], coords[ps, 1], c='#2C3E50', s=130, marker='s', zorder=5,
                   edgecolors='white', linewidths=1.5)

    # 中心站A
    ax.scatter(coords[0, 0], coords[0, 1], c='#E74C3C', s=350, marker='*', zorder=6)
    ax.annotate('A', xy=(26, 31), xytext=(27, 29), fontsize=13, fontweight='bold', color='#E74C3C')

    legend_elements = [
        Line2D([0], [0], color='#E74C3C', linewidth=2.5, label=f'I型管道 ({L_I:.2f} km)'),
        Line2D([0], [0], color='#3498DB', linewidth=1.0, label=f'II型管道 ({L_II:.2f} km)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#E74C3C', markersize=15, label='中心站 A'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2C3E50', markersize=10, label='一级供水站'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=7, label='二级供水站'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    ax.set_title(f'图5  最优输水管道网络（总费用 {C_total/10000:.2f} 万元）', fontsize=14)
    ax.set_xlabel('X 坐标 (km)', fontsize=12)
    ax.set_ylabel('Y 坐标 (km)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/图5_完整管网拓扑.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/图5_完整管网拓扑.pdf', bbox_inches='tight')
    plt.close()
    print("  图5 已保存")


def plot_fig6(cluster_mst_lengths):
    """图6：各簇II型管道长度柱状图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    K = len(cluster_mst_lengths)
    x = np.arange(1, K + 1)

    bars = ax.bar(x, cluster_mst_lengths, color=plt.cm.Blues(np.linspace(0.4, 0.9, K)),
                  edgecolor='white', linewidth=0.5)
    ax.axhline(y=30.0, color='#E74C3C', linestyle='--', linewidth=2, label='30 km 约束上限')

    for bar, h in zip(bars, cluster_mst_lengths):
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    ax.set_xlabel('簇编号', fontsize=12)
    ax.set_ylabel('II型管道MST总长 (km)', fontsize=12)
    ax.set_title('图6  各簇II型管道长度与30km约束', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in x])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 35)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/图6_各簇管道长度.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/图6_各簇管道长度.pdf', bbox_inches='tight')
    plt.close()
    print("  图6 已保存")


def plot_fig7(convergence_curve, initial_cost):
    """图7：模拟退火收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    iterations = [item[0] for item in convergence_curve]
    costs = [item[1] / 10000 for item in convergence_curve]

    ax.plot(iterations, costs, c='#2C3E50', linewidth=1.5)
    ax.scatter([iterations[0]], [costs[0]], c='#E74C3C', s=80, zorder=5,
               label=f'初始解: {costs[0]:.2f} 万元')
    ax.scatter([iterations[-1]], [costs[-1]], c='#27AE60', s=80, zorder=5,
               label=f'最优解: {costs[-1]:.2f} 万元')

    improvement = (costs[0] - costs[-1]) / costs[0] * 100
    if len(iterations) > 1:
        ax.annotate(f'改善 {improvement:.1f}%',
                    xy=(iterations[-1], costs[-1]),
                    xytext=(iterations[-1] * 0.6, costs[-1] * 1.03),
                    fontsize=11, color='#27AE60', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#27AE60'))

    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('总费用 (万元)', fontsize=12)
    ax.set_title('图7  模拟退火收敛曲线', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/图7_收敛曲线.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/图7_收敛曲线.pdf', bbox_inches='tight')
    plt.close()
    print("  图7 已保存")


def plot_fig8(L_I, L_II, C_total):
    """图8：成本构成分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    type_labels = ['I型管道\n(1291元/m)', 'II型管道\n(445元/m)']
    type_costs = [1291000 * L_I, 445000 * L_II]
    colors_type = ['#E74C3C', '#3498DB']
    ax1.pie(type_costs, labels=type_labels, colors=colors_type, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax1.set_title('按管道类型', fontsize=13)

    detail_labels = ['I型材料费\n(571元/m)', 'I型铺设费\n(720元/m)',
                     'II型材料费\n(235元/m)', 'II型铺设费\n(210元/m)']
    detail_costs = [571000 * L_I, 720000 * L_I, 235000 * L_II, 210000 * L_II]
    colors_detail = ['#E74C3C', '#F1948A', '#3498DB', '#85C1E9']
    ax2.pie(detail_costs, labels=detail_labels, colors=colors_detail, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('按费用性质', fontsize=13)

    fig.suptitle(f'图8  总费用构成分析（总计 {C_total/10000:.2f} 万元）', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/图8_成本构成.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/图8_成本构成.pdf', bbox_inches='tight')
    plt.close()
    print("  图8 已保存")


def generate_all_figures(coords, dist_matrix, clusters, primary_stations,
                         cluster_mst_lengths, cluster_mst_edges,
                         E_I, E_II, L_I, L_II, C_total, convergence_curve, initial_cost):
    """生成全部8张图"""
    print("[可视化] 开始生成论文配图...")
    plot_fig1(coords)
    plot_fig2(coords, dist_matrix)
    plot_fig3(coords, clusters, primary_stations, cluster_mst_lengths, cluster_mst_edges)
    plot_fig4(coords, E_I, primary_stations, L_I, dist_matrix)
    plot_fig5(coords, E_I, E_II, primary_stations, L_I, L_II, C_total)
    plot_fig6(cluster_mst_lengths)
    plot_fig7(convergence_curve, initial_cost)
    plot_fig8(L_I, L_II, C_total)
    print("[可视化] 全部8张图生成完毕")

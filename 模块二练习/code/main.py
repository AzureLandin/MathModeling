# -*- coding: utf-8 -*-
"""
B题 管道最优铺设问题 — 主程序
串联五步求解流程，输出CSV结果与论文配图
"""

import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import preprocess
from clustering import constrained_agglomerative_clustering
from station_selection import select_primary_stations
from network_design import build_network, C_I, C_II
from optimization import simulated_annealing, Solution, evaluate_solution
from visualization import generate_all_figures

RESULTS_DIR = r'E:\MathModeling\模块二练习\results'


def save_results_csv(solution, cluster_mst_lengths, convergence_curve, initial_cost):
    """保存计算结果为CSV文件"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. 总览结果
    with open(f'{RESULTS_DIR}/结果总览.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['指标', '数值', '单位'])
        writer.writerow(['一级供水站数量', len(solution.primary_stations), '个'])
        writer.writerow(['一级站编号', ';'.join(map(str, sorted(solution.primary_stations))), ''])
        writer.writerow(['I型管道边数', len(solution.E_I), '条'])
        writer.writerow(['I型管道总长', f'{solution.L_I:.6f}', '公里'])
        writer.writerow(['I型管道费用', f'{C_I * solution.L_I:.2f}', '元'])
        writer.writerow(['II型管道边数', sum(len(e) for e in solution.E_II), '条'])
        writer.writerow(['II型管道总长', f'{solution.L_II:.6f}', '公里'])
        writer.writerow(['II型管道费用', f'{C_II * solution.L_II:.2f}', '元'])
        writer.writerow(['总费用', f'{solution.C_total:.2f}', '元'])
        writer.writerow(['总费用', f'{solution.C_total/10000:.4f}', '万元'])
        writer.writerow(['初始解费用', f'{initial_cost:.2f}', '元'])
        writer.writerow(['优化改善率', f'{(initial_cost - solution.C_total)/initial_cost*100:.2f}', '%'])
    print(f"  结果总览.csv 已保存")

    # 2. 各簇详情
    with open(f'{RESULTS_DIR}/各簇详情.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['簇编号', '一级站编号', '簇内节点数', 'II型管道长度_km',
                         '是否满足30km约束', '簇内节点列表'])
        for k in range(len(solution.clusters)):
            nodes = sorted(solution.clusters[k])
            writer.writerow([
                k + 1,
                solution.primary_stations[k],
                len(nodes),
                f'{cluster_mst_lengths[k]:.6f}',
                '是' if cluster_mst_lengths[k] <= 30.0 else '否',
                ';'.join(map(str, nodes))
            ])
    print(f"  各簇详情.csv 已保存")

    # 3. I型管道边列表
    with open(f'{RESULTS_DIR}/I型管道边.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['边序号', '起点', '终点', '长度_km', '费用_元'])
        for idx, (u, v) in enumerate(solution.E_I, 1):
            length = np.sqrt((u - v) ** 2) if False else 0  # placeholder
            # 用dist_matrix计算
            writer.writerow([idx, u, v, '', ''])
    # 重新用dist_matrix写
    print(f"  I型管道边.csv 已保存")

    # 4. II型管道边列表
    with open(f'{RESULTS_DIR}/II型管道边.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['簇编号', '边序号', '起点', '终点'])
        for k, edges in enumerate(solution.E_II, 1):
            for idx, (u, v) in enumerate(edges, 1):
                writer.writerow([k, idx, u, v])
    print(f"  II型管道边.csv 已保存")

    # 5. 收敛曲线数据
    with open(f'{RESULTS_DIR}/收敛曲线.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['迭代次数', '总费用_元', '总费用_万元'])
        for it, cost in convergence_curve:
            writer.writerow([it, f'{cost:.2f}', f'{cost/10000:.4f}'])
    print(f"  收敛曲线.csv 已保存")


def save_detailed_edges(solution, dist_matrix):
    """保存带距离和费用的管道边详情"""
    # I型管道
    with open(f'{RESULTS_DIR}/I型管道边.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['边序号', '起点', '终点', '长度_km', '费用_元'])
        for idx, (u, v) in enumerate(solution.E_I, 1):
            length = dist_matrix[u][v]
            cost = C_I * length
            writer.writerow([idx, u, v, f'{length:.6f}', f'{cost:.2f}'])

    # II型管道
    with open(f'{RESULTS_DIR}/II型管道边.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['簇编号', '边序号', '起点', '终点', '长度_km', '费用_元'])
        for k, edges in enumerate(solution.E_II, 1):
            for idx, (u, v) in enumerate(edges, 1):
                length = dist_matrix[u][v]
                cost = C_II * length
                writer.writerow([k, idx, u, v, f'{length:.6f}', f'{cost:.2f}'])


def main():
    print("=" * 60)
    print("B题 管道最优铺设问题 — 求解程序")
    print("=" * 60)

    # ===== 第一步：数据预处理 =====
    coords, dist_matrix, delaunay_edges = preprocess()

    # ===== 第二步：约束聚类 =====
    clusters, cluster_mst_lengths, cluster_mst_edges = constrained_agglomerative_clustering(
        nodes=list(range(1, 181)), dist_matrix=dist_matrix, max_mst_length=30.0
    )

    # ===== 第三步：一级站选址 =====
    primary_stations, S, T = select_primary_stations(
        clusters=clusters, dist_matrix=dist_matrix, center_node=0
    )

    # ===== 第四步：管网生成 =====
    E_I, E_II, L_I, L_II, C_total, cluster_lengths = build_network(
        clusters, primary_stations, dist_matrix
    )
    initial_cost = C_total

    # ===== 第五步：模拟退火优化 =====
    X0 = Solution(clusters, primary_stations, E_I, E_II, L_I, L_II, C_total)
    print(f"\n[第五步] 开始模拟退火优化 (初始费用: {C_total/10000:.2f}万元)...")
    X_best, convergence_curve = simulated_annealing(X0, dist_matrix, n_restarts=5)

    # 用最优解重新计算簇MST信息
    _, _, final_cluster_lengths = evaluate_solution(
        X_best.clusters, X_best.primary_stations, dist_matrix
    )[:3]
    # 重新获取完整信息
    from network_design import generate_secondary_network
    E_II_final, L_II_final, cluster_mst_final = generate_secondary_network(
        X_best.clusters, dist_matrix
    )
    # 重新计算MST边
    from utils import compute_mst
    cluster_mst_edges_final = []
    for cluster in X_best.clusters:
        _, edges = compute_mst(cluster, dist_matrix)
        cluster_mst_edges_final.append(edges)

    # ===== 保存CSV结果 =====
    print(f"\n[输出] 保存计算结果...")
    save_results_csv(X_best, cluster_mst_final, convergence_curve, initial_cost)
    save_detailed_edges(X_best, dist_matrix)

    # ===== 可视化 =====
    generate_all_figures(
        coords=coords, dist_matrix=dist_matrix,
        clusters=X_best.clusters, primary_stations=X_best.primary_stations,
        cluster_mst_lengths=cluster_mst_final, cluster_mst_edges=cluster_mst_edges_final,
        E_I=X_best.E_I, E_II=X_best.E_II,
        L_I=X_best.L_I, L_II=X_best.L_II, C_total=X_best.C_total,
        convergence_curve=convergence_curve, initial_cost=initial_cost
    )

    # ===== 最终摘要 =====
    print("\n" + "=" * 60)
    print("求解完成！最终结果摘要：")
    print(f"  一级供水站: {len(X_best.primary_stations)}个")
    print(f"  I型管道总长: {X_best.L_I:.4f} km")
    print(f"  II型管道总长: {X_best.L_II:.4f} km")
    print(f"  总费用: {X_best.C_total:.2f} 元 ({X_best.C_total/10000:.4f} 万元)")
    print(f"  相比初始解改善: {(initial_cost - X_best.C_total)/initial_cost*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

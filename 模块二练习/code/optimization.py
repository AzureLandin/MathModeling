# -*- coding: utf-8 -*-
"""
第五步：模拟退火优化
"""

import numpy as np
import copy
import random
from utils import compute_mst, compute_mst_length
from network_design import C_I, C_II, generate_secondary_network, generate_primary_network, calculate_total_cost
from station_selection import select_primary_stations


class Solution:
    """解的编码"""
    def __init__(self, clusters, primary_stations, E_I, E_II, L_I, L_II, C_total):
        self.clusters = clusters
        self.primary_stations = primary_stations
        self.E_I = E_I
        self.E_II = E_II
        self.L_I = L_I
        self.L_II = L_II
        self.C_total = C_total

    def copy(self):
        return Solution(
            clusters=[set(c) for c in self.clusters],
            primary_stations=list(self.primary_stations),
            E_I=list(self.E_I),
            E_II=[list(e) for e in self.E_II],
            L_I=self.L_I,
            L_II=self.L_II,
            C_total=self.C_total
        )


def evaluate_solution(clusters, primary_stations, dist_matrix):
    """评估一个解的总成本"""
    E_II, L_II, cluster_lengths = generate_secondary_network(clusters, dist_matrix)
    E_I, L_I = generate_primary_network(primary_stations, dist_matrix)
    C_total, _, _ = calculate_total_cost(L_I, L_II)
    return E_I, E_II, L_I, L_II, C_total, cluster_lengths


def operator_node_reassign(solution, dist_matrix, max_mst=30.0):
    """算子A：边界节点漂移"""
    clusters = [set(c) for c in solution.clusters]
    K = len(clusters)
    if K <= 1:
        return None

    # 随机选一个非一级站的节点
    all_secondary = []
    for k, c in enumerate(clusters):
        ps = solution.primary_stations[k]
        for v in c:
            if v != ps:
                all_secondary.append((v, k))

    if not all_secondary:
        return None

    v, src_k = random.choice(all_secondary)

    # 找最近的另一个簇
    best_target = None
    best_dist = float('inf')
    for k in range(K):
        if k == src_k:
            continue
        for u in clusters[k]:
            if dist_matrix[v][u] < best_dist:
                best_dist = dist_matrix[v][u]
                best_target = k

    if best_target is None:
        return None

    # 尝试移动
    clusters[src_k].remove(v)
    clusters[best_target].add(v)

    # 校验目标簇MST
    if compute_mst_length(clusters[best_target], dist_matrix) > max_mst:
        return None

    # 源簇不能为空
    if len(clusters[src_k]) == 0:
        return None

    # 重新选一级站
    new_ps = list(solution.primary_stations)
    # 如果移走的是一级站（不会，因为排除了），或者目标簇一级站需更新
    if new_ps[best_target] not in clusters[best_target]:
        # 选簇内到A最近的
        new_ps[best_target] = min(clusters[best_target], key=lambda n: dist_matrix[n][0])
    if new_ps[src_k] not in clusters[src_k]:
        new_ps[src_k] = min(clusters[src_k], key=lambda n: dist_matrix[n][0])

    # 评估
    E_I, E_II, L_I, L_II, C_total, _ = evaluate_solution(clusters, new_ps, dist_matrix)
    return Solution(clusters, new_ps, E_I, E_II, L_I, L_II, C_total)


def operator_primary_swap(solution, dist_matrix):
    """算子B：一级站替换"""
    clusters = [set(c) for c in solution.clusters]
    K = len(clusters)

    k = random.randint(0, K - 1)
    candidates = [v for v in clusters[k] if v != solution.primary_stations[k]]
    if not candidates:
        return None

    new_v = random.choice(candidates)
    new_ps = list(solution.primary_stations)
    new_ps[k] = new_v

    E_I, E_II, L_I, L_II, C_total, _ = evaluate_solution(clusters, new_ps, dist_matrix)
    return Solution(clusters, new_ps, E_I, E_II, L_I, L_II, C_total)


def operator_split_merge(solution, dist_matrix, max_mst=30.0):
    """算子C：簇分裂与合并"""
    clusters = [set(c) for c in solution.clusters]
    K = len(clusters)

    # 50%概率分裂，50%概率合并
    if random.random() < 0.5 and K >= 2:
        # 合并：随机选两个簇
        k1, k2 = random.sample(range(K), 2)
        merged = clusters[k1] | clusters[k2]
        if compute_mst_length(merged, dist_matrix) > max_mst:
            return None

        new_clusters = [clusters[i] for i in range(K) if i != k1 and i != k2]
        new_clusters.append(merged)

        # 重新选一级站
        new_ps_list = [solution.primary_stations[i] for i in range(K) if i != k1 and i != k2]
        # 合并簇的一级站：选两个原一级站中更优的
        ps1 = solution.primary_stations[k1]
        ps2 = solution.primary_stations[k2]
        d1 = dist_matrix[ps1][0]
        d2 = dist_matrix[ps2][0]
        new_ps_list.append(ps1 if d1 < d2 else ps2)

        E_I, E_II, L_I, L_II, C_total, _ = evaluate_solution(new_clusters, new_ps_list, dist_matrix)
        return Solution(new_clusters, new_ps_list, E_I, E_II, L_I, L_II, C_total)

    else:
        # 分裂：选节点数>=4的簇
        big_clusters = [k for k in range(K) if len(clusters[k]) >= 4]
        if not big_clusters:
            return None

        k = random.choice(big_clusters)
        nodes = list(clusters[k])
        random.shuffle(nodes)
        split_size = max(2, len(nodes) // 3)

        sub1 = set(nodes[:split_size])
        sub2 = set(nodes[split_size:])

        if len(sub2) == 0:
            return None

        # 校验两个子簇MST
        if compute_mst_length(sub1, dist_matrix) > max_mst:
            return None
        if compute_mst_length(sub2, dist_matrix) > max_mst:
            return None

        new_clusters = [clusters[i] for i in range(K) if i != k]
        new_clusters.append(sub1)
        new_clusters.append(sub2)

        new_ps_list = [solution.primary_stations[i] for i in range(K) if i != k]
        # 为两个子簇选一级站
        ps_orig = solution.primary_stations[k]
        if ps_orig in sub1:
            new_ps_list.append(ps_orig)
            new_ps_list.append(min(sub2, key=lambda n: dist_matrix[n][0]))
        else:
            new_ps_list.append(min(sub1, key=lambda n: dist_matrix[n][0]))
            new_ps_list.append(ps_orig if ps_orig in sub2 else min(sub2, key=lambda n: dist_matrix[n][0]))

        E_I, E_II, L_I, L_II, C_total, _ = evaluate_solution(new_clusters, new_ps_list, dist_matrix)
        return Solution(new_clusters, new_ps_list, E_I, E_II, L_I, L_II, C_total)


def simulated_annealing(X0, dist_matrix, n_restarts=5, seed_base=42):
    """
    模拟退火主框架（含多次重启）
    """
    best_overall = X0
    all_convergence = []

    for restart in range(n_restarts):
        random.seed(seed_base + restart)
        np.random.seed(seed_base + restart)

        X_current = X0.copy()
        X_best = X0.copy()
        convergence_curve = [(0, X0.C_total)]

        # 自适应初始温度
        deltas = []
        for _ in range(50):
            r = random.random()
            if r < 0.5:
                X_n = operator_node_reassign(X_current, dist_matrix)
            elif r < 0.8:
                X_n = operator_primary_swap(X_current, dist_matrix)
            else:
                X_n = operator_split_merge(X_current, dist_matrix)
            if X_n is not None:
                deltas.append(X_n.C_total - X_current.C_total)

        positive_deltas = [d for d in deltas if d > 0]
        if positive_deltas:
            T = -np.mean(positive_deltas) / np.log(0.8)
        else:
            T = X0.C_total * 0.01

        T_end = 0.01
        alpha_cool = 0.995
        n_iter_per_temp = 100
        no_improve_count = 0
        total_iter = 0

        while T > T_end and no_improve_count < 3000 and total_iter < 30000:
            for _ in range(n_iter_per_temp):
                total_iter += 1
                r = random.random()
                if r < 0.5:
                    X_neighbor = operator_node_reassign(X_current, dist_matrix)
                elif r < 0.8:
                    X_neighbor = operator_primary_swap(X_current, dist_matrix)
                else:
                    X_neighbor = operator_split_merge(X_current, dist_matrix)

                if X_neighbor is None:
                    continue

                delta_C = X_neighbor.C_total - X_current.C_total

                # Metropolis准则
                if delta_C < 0:
                    X_current = X_neighbor
                else:
                    if T > 0 and random.random() < np.exp(-delta_C / T):
                        X_current = X_neighbor

                if X_current.C_total < X_best.C_total:
                    X_best = X_current.copy()
                    convergence_curve.append((total_iter, X_best.C_total))
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            T *= alpha_cool

        all_convergence.append(convergence_curve)
        print(f"  重启{restart+1}: 最优={X_best.C_total/10000:.2f}万元, "
              f"迭代{total_iter}次, 一级站{len(X_best.primary_stations)}个")

        if X_best.C_total < best_overall.C_total:
            best_overall = X_best

    # 合并收敛曲线（取最优重启的）
    best_convergence = min(all_convergence, key=lambda c: c[-1][1])
    print(f"[第五步] 模拟退火完成: 最优总费用={best_overall.C_total/10000:.2f}万元, "
          f"改善{(X0.C_total - best_overall.C_total)/X0.C_total*100:.2f}%")

    return best_overall, best_convergence

# -*- coding: utf-8 -*-
"""
第四步：全局图论组网与总成本核算
"""

import numpy as np
from utils import compute_mst

# 管道单价（元/公里）
C_I = 1291000   # I型: 材料571 + 铺设720 = 1291元/米 = 1291000元/公里
C_II = 445000   # II型: 材料235 + 铺设210 = 445元/米 = 445000元/公里


def generate_secondary_network(clusters, dist_matrix):
    """
    生成II型管道网络：每个簇内MST
    OUTPUT: E_II - 各簇II型管道边列表[list of list[tuple]]
            L_II - II型管道总长(km)
            cluster_lengths - 各簇MST长度
    """
    E_II = []
    L_II = 0.0
    cluster_lengths = []

    for cluster in clusters:
        length, edges = compute_mst(cluster, dist_matrix)
        E_II.append(edges)
        L_II += length
        cluster_lengths.append(length)

    return E_II, L_II, cluster_lengths


def generate_primary_network(primary_stations, dist_matrix, center_node=0):
    """
    生成I型管道网络：中心站A + 所有一级站的MST
    OUTPUT: E_I - I型管道边列表
            L_I - I型管道总长(km)
    """
    nodes_I = set(primary_stations) | {center_node}
    L_I, E_I = compute_mst(nodes_I, dist_matrix)
    return E_I, L_I


def calculate_total_cost(L_I, L_II):
    """计算总费用"""
    cost_I = C_I * L_I
    cost_II = C_II * L_II
    return cost_I + cost_II, cost_I, cost_II


def build_network(clusters, primary_stations, dist_matrix):
    """第四步主函数：组网 + 成本核算"""
    E_II, L_II, cluster_lengths = generate_secondary_network(clusters, dist_matrix)
    E_I, L_I = generate_primary_network(primary_stations, dist_matrix)
    C_total, cost_I, cost_II = calculate_total_cost(L_I, L_II)

    print(f"[第四步] 管网生成完成:")
    print(f"  I型管道: {len(E_I)}条边, 总长{L_I:.4f}km, 费用{cost_I/10000:.2f}万元")
    print(f"  II型管道: {sum(len(e) for e in E_II)}条边, 总长{L_II:.4f}km, 费用{cost_II/10000:.2f}万元")
    print(f"  总费用: {C_total/10000:.2f}万元")

    return E_I, E_II, L_I, L_II, C_total, cluster_lengths

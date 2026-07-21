# -*- coding: utf-8 -*-
"""
第三步：确定一级供水站的最优位置
"""

import numpy as np


def select_primary_stations(clusters, dist_matrix, center_node=0, alpha=0.6, beta=0.4):
    """
    加权评分选址：按簇质心到A距离从近到远依次处理
    INPUT:  clusters - 簇列表
            dist_matrix - 距离矩阵
            center_node - 中心站节点编号(0)
            alpha - 内部成本权重
            beta - 外部成本权重
    OUTPUT: primary_stations - 每簇一级站编号列表
    """
    # 按簇质心到A的距离排序
    centroids = []
    for cluster in clusters:
        nodes = list(cluster)
        cx = np.mean([dist_matrix[n][0] for n in nodes])  # 用dist_matrix[n][0]不对，应该用坐标
        # 这里用节点到A的距离均值作为排序依据
        avg_dist_to_A = np.mean([dist_matrix[n][center_node] for n in nodes])
        centroids.append(avg_dist_to_A)

    sorted_indices = np.argsort(centroids)  # 从近到远

    primary_stations = [None] * len(clusters)
    S_already = set()

    for idx in sorted_indices:
        cluster = clusters[idx]
        best_node = None
        best_score = float('inf')

        for v in cluster:
            # 内部成本：到簇内所有其他节点的距离之和
            f_int = sum(dist_matrix[v][u] for u in cluster if u != v)

            # 外部成本：到A或已有一级站的最近距离
            f_ext = dist_matrix[v][center_node]
            if S_already:
                f_ext = min(f_ext, min(dist_matrix[v][s] for s in S_already))

            score = alpha * f_int + beta * f_ext
            if score < best_score:
                best_score = score
                best_node = v

        primary_stations[idx] = best_node
        S_already.add(best_node)

    S = set(primary_stations)
    T = set(range(1, 181)) - S
    print(f"[第三步] 一级站选址完成: {len(S)}个一级站, 编号={sorted(S)}")

    return primary_stations, S, T

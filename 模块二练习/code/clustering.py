# -*- coding: utf-8 -*-
"""
第二步：基于30公里MST上限的约束凝聚聚类
"""

import numpy as np
from utils import compute_mst, compute_mst_length


def constrained_agglomerative_clustering(nodes, dist_matrix, max_mst_length=30.0):
    """
    约束凝聚聚类：单链接准则 + MST长度校验
    INPUT:  nodes - 待聚类节点列表(1-180)
            dist_matrix - 距离矩阵
            max_mst_length - 每簇MST上限(30km)
    OUTPUT: clusters - 簇列表[list of set]
            cluster_mst_lengths - 各簇MST长度
            cluster_mst_edges - 各簇MST边列表
    """
    # 初始化：每个节点自成一簇
    clusters = [frozenset([n]) for n in nodes]
    cluster_mst = {i: 0.0 for i in range(len(clusters))}

    # 不可合并对集合
    forbidden_pairs = set()

    def single_link_distance(c1, c2):
        """单链接：两簇间最近节点距离"""
        min_d = float('inf')
        for i in c1:
            for j in c2:
                if dist_matrix[i][j] < min_d:
                    min_d = dist_matrix[i][j]
        return min_d

    iteration = 0
    while True:
        # 找距离最近的合法簇对
        best_pair = None
        best_dist = float('inf')

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if (i, j) in forbidden_pairs:
                    continue
                d = single_link_distance(clusters[i], clusters[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        if best_pair is None:
            break  # 所有簇对都不可合并

        i, j = best_pair
        # 尝试合并
        merged = clusters[i] | clusters[j]
        mst_len = compute_mst_length(merged, dist_matrix)

        if mst_len <= max_mst_length:
            # 合并成功
            new_clusters = []
            for k in range(len(clusters)):
                if k != i and k != j:
                    new_clusters.append(clusters[k])
            new_clusters.append(merged)
            clusters = new_clusters
            # 重置forbidden_pairs（索引变了）
            forbidden_pairs = set()
            iteration += 1
        else:
            # 合并失败，标记禁止
            forbidden_pairs.add((i, j))

    # 计算最终各簇的MST信息
    cluster_mst_lengths = []
    cluster_mst_edges = []
    for cluster in clusters:
        length, edges = compute_mst(cluster, dist_matrix)
        cluster_mst_lengths.append(length)
        cluster_mst_edges.append(edges)

    print(f"[第二步] 约束聚类完成: {len(clusters)}个簇, "
          f"MST长度范围[{min(cluster_mst_lengths):.2f}, {max(cluster_mst_lengths):.2f}]km")

    return clusters, cluster_mst_lengths, cluster_mst_edges

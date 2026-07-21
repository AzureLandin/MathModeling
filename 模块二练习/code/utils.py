# -*- coding: utf-8 -*-
"""
公共工具模块：MST计算、距离计算等
"""

import numpy as np
from scipy.spatial.distance import cdist


def compute_distance_matrix(coords):
    """计算全连接欧氏距离矩阵"""
    return cdist(coords, coords, metric='euclidean')


def compute_mst(node_subset, dist_matrix):
    """
    Prim算法计算给定节点子集的MST
    返回: (总长度, 边列表[(u,v), ...])
    """
    nodes = list(node_subset)
    n = len(nodes)
    if n <= 1:
        return 0.0, []

    in_tree = {nodes[0]}
    mst_edges = []
    total_length = 0.0

    while len(in_tree) < n:
        min_edge = None
        min_dist = float('inf')
        for u in in_tree:
            for v in nodes:
                if v not in in_tree and dist_matrix[u][v] < min_dist:
                    min_dist = dist_matrix[u][v]
                    min_edge = (u, v)
        if min_edge is None:
            break
        in_tree.add(min_edge[1])
        mst_edges.append(min_edge)
        total_length += min_dist

    return total_length, mst_edges


def compute_mst_length(node_subset, dist_matrix):
    """仅计算MST总长度（不返回边列表，用于聚类校验）"""
    nodes = list(node_subset)
    n = len(nodes)
    if n <= 1:
        return 0.0

    in_tree = {nodes[0]}
    total_length = 0.0

    while len(in_tree) < n:
        min_dist = float('inf')
        min_node = None
        for u in in_tree:
            for v in nodes:
                if v not in in_tree and dist_matrix[u][v] < min_dist:
                    min_dist = dist_matrix[u][v]
                    min_node = v
        if min_node is None:
            break
        in_tree.add(min_node)
        total_length += min_dist

    return total_length


def euclidean_distance(p1, p2):
    """两点间欧氏距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

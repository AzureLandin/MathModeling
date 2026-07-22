# -*- coding: utf-8 -*-
"""根据已保存的最终结果快速重绘论文图片，不重复执行模拟退火。"""

import csv
import os

from data_loader import preprocess
from visualization import generate_all_figures


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def read_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():
    coords, dist_matrix, _ = preprocess()

    cluster_rows = read_csv("各簇详情.csv")
    clusters = [set(map(int, row["簇内节点列表"].split(";"))) for row in cluster_rows]
    primary_stations = [int(row["一级站编号"]) for row in cluster_rows]
    cluster_mst_lengths = [float(row["II型管道长度_km"]) for row in cluster_rows]

    primary_rows = read_csv("I型管道边.csv")
    E_I = [(int(row["起点"]), int(row["终点"])) for row in primary_rows]

    E_II = [[] for _ in clusters]
    for row in read_csv("II型管道边.csv"):
        E_II[int(row["簇编号"]) - 1].append((int(row["起点"]), int(row["终点"])))

    convergence_rows = read_csv("收敛曲线.csv")
    convergence_curve = [
        (int(row["迭代次数"]), float(row["总费用_元"]))
        for row in convergence_rows
    ]

    summary_rows = read_csv("结果总览.csv")
    summary = {}
    for row in summary_rows:
        summary.setdefault(row["指标"], []).append(float(row["数值"]) if row["指标"] != "一级站编号" else row["数值"])

    L_I = summary["I型管道总长"][0]
    L_II = summary["II型管道总长"][0]
    C_total = next(float(row["数值"]) for row in summary_rows if row["指标"] == "总费用" and row["单位"] == "元")
    initial_cost = summary["初始解费用"][0]

    generate_all_figures(
        coords=coords,
        dist_matrix=dist_matrix,
        clusters=clusters,
        primary_stations=primary_stations,
        cluster_mst_lengths=cluster_mst_lengths,
        cluster_mst_edges=E_II,
        E_I=E_I,
        E_II=E_II,
        L_I=L_I,
        L_II=L_II,
        C_total=C_total,
        convergence_curve=convergence_curve,
        initial_cost=initial_cost,
    )


if __name__ == "__main__":
    main()

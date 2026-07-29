"""
问题二求解器模块：0-1整数规划求解最优订购方案
"""
import math
import numpy as np
import pulp


def load_top55_params(top55_path, order_data, supplier_ids):
    """
    从问题一结果中提取55家供应商的参数
    
    返回:
        indices:   55家供应商在原始数据中的索引
        Q:         固定订货量（四舍五入取整）
        F:         最低订货次数
        u:         消耗率
        p:         相对单价
        ids:       供应商ID
        types:     材料类别
    """
    import pandas as pd
    top55 = pd.read_csv(top55_path)
    
    # 材料类型映射
    type_map = {'A': (0.6, 1.2), 'B': (0.66, 1.1), 'C': (0.72, 1.0)}
    
    indices = []
    Q = []
    F = []
    u = []
    p = []
    ids = []
    types = []
    
    for _, row in top55.iterrows():
        sid = row['供应商ID']
        mat_type = row['材料类别']
        
        # 找到在原始数据中的索引
        idx = np.where(supplier_ids == sid)[0][0]
        indices.append(idx)
        
        # Q_i: 周均订货量四舍五入取整
        valid = order_data[idx, :] > 0
        N_i = valid.sum()
        if N_i > 0:
            q_val = order_data[idx, valid].mean()
            Q_i = int(q_val + 0.5)  # 标准四舍五入
        else:
            Q_i = 0
        Q.append(Q_i)
        
        # F_i: 平均周期订货次数（向上取整）
        F_i = math.ceil(N_i / 10)
        F.append(F_i)
        
        # u_i, p_i
        u_i, p_i = type_map[mat_type]
        u.append(u_i)
        p.append(p_i)
        
        ids.append(sid)
        types.append(mat_type)
    
    return indices, np.array(Q), np.array(F), np.array(u), np.array(p), ids, types


def check_feasibility(Q, u, F):
    """
    检查问题可行性
    """
    # 55家全选时的周产能
    total_capacity = np.sum(Q / u)
    min_required = 25380  # 28200 * 0.9
    
    print(f"\n=== 可行性检查 ===")
    print(f"  55家全选周产能: {total_capacity:.0f} m³")
    print(f"  最低要求: {min_required} m³")
    print(f"  产能余量: {total_capacity - min_required:.0f} m³")
    print(f"  频次约束总和: {F.sum()}")
    print(f"  频次约束上限: {55 * 24} = {55*24}")
    
    if total_capacity < min_required:
        print("  ⚠️ 警告：55家全选仍不满足产能要求！")
        return False
    
    print("  ✓ 可行性检查通过")
    return True


def solve_ordering_problem(Q, u, p, F):
    """
    求解0-1整数规划问题
    
    返回:
        x:       55×24的0-1决策矩阵
        status:  求解状态
        obj_val: 最优目标值
        time:    求解时间
    """
    n = len(Q)  # 55
    T = 24      # 24周
    
    print("\n正在构建优化模型...")
    
    # 创建问题
    prob = pulp.LpProblem("RawMaterialOrdering", pulp.LpMinimize)
    
    # 决策变量
    x = {}
    for i in range(n):
        for j in range(T):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
    
    # 目标函数：最小化总成本
    prob += pulp.lpSum(p[i] * Q[i] * x[i, j] for i in range(n) for j in range(T))
    
    # 约束一：周产能底线约束（24个）
    for j in range(T):
        prob += pulp.lpSum(Q[i] / u[i] * x[i, j] for i in range(n)) >= 25380
    
    # 约束二：最低订购频次约束（55个）
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(T)) >= F[i]
    
    print("正在求解...")
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    
    # 提取结果
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    
    # 构建决策矩阵
    result = np.zeros((n, T), dtype=int)
    for i in range(n):
        for j in range(T):
            result[i, j] = int(x[i, j].varValue)
    
    return result, status, obj_val


def compute_weekly_summary(x, Q, u, p, types):
    """
    计算每周汇总统计
    
    返回:
        weekly_volume:  每周总订购量（产品体积）
        weekly_cost:    每周总成本
        weekly_count:   每周选中供应商数
        weekly_abc:     每周A/B/C类型数量
    """
    T = 24
    weekly_volume = np.zeros(T)
    weekly_cost = np.zeros(T)
    weekly_count = np.zeros(T, dtype=int)
    weekly_A = np.zeros(T, dtype=int)
    weekly_B = np.zeros(T, dtype=int)
    weekly_C = np.zeros(T, dtype=int)
    
    for j in range(T):
        for i in range(len(Q)):
            if x[i, j] == 1:
                weekly_volume[j] += Q[i] / u[i]
                weekly_cost[j] += p[i] * Q[i]
                weekly_count[j] += 1
                if types[i] == 'A':
                    weekly_A[j] += 1
                elif types[i] == 'B':
                    weekly_B[j] += 1
                else:
                    weekly_C[j] += 1
    
    return weekly_volume, weekly_cost, weekly_count, weekly_A, weekly_B, weekly_C

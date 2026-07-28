"""
数据读取模块：负责加载Excel数据并提取订货量/供货量矩阵
"""
import numpy as np
import pandas as pd


def load_supplier_data(file_path):
    """
    读取附件1的订货量和供货量数据
    
    参数:
        file_path: Excel文件路径
    
    返回:
        supplier_ids: 供应商ID数组
        material_types: 材料类别数组
        order_data: 订货量矩阵 (402×240)
        supply_data: 供货量矩阵 (402×240)
    """
    print("正在读取数据...")
    order_df = pd.read_excel(file_path, sheet_name=0)   # 订货量
    supply_df = pd.read_excel(file_path, sheet_name=1)   # 供货量

    supplier_ids = order_df.iloc[:, 0].values
    material_types = order_df.iloc[:, 1].values
    order_data = order_df.iloc[:, 2:242].values.astype(float)
    supply_data = supply_df.iloc[:, 2:242].values.astype(float)

    print(f"  供应商数量: {len(supplier_ids)}")
    print(f"  数据维度: {order_data.shape}")

    return supplier_ids, material_types, order_data, supply_data

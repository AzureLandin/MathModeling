"""
数据读取模块
"""
import numpy as np
import pandas as pd


def load_supplier_data(file_path):
    print("正在读取数据...")
    order_df = pd.read_excel(file_path, sheet_name=0)
    supply_df = pd.read_excel(file_path, sheet_name=1)

    supplier_ids = order_df.iloc[:, 0].values
    material_types = order_df.iloc[:, 1].values
    order_data = order_df.iloc[:, 2:242].values.astype(float)
    supply_data = supply_df.iloc[:, 2:242].values.astype(float)

    print(f"  供应商数量: {len(supplier_ids)}")
    print(f"  数据维度: {order_data.shape}")

    return supplier_ids, material_types, order_data, supply_data

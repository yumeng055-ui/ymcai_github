"""
自动化端到端流水线：HDB 组屋价格预测
说明：直接在根目录读取数据并生成 .pkl 模型文件
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# ================= 全局配置 =================
RANDOM_SEED = 64

# 获取当前脚本运行的根目录
ROOT_DIR = Path.cwd()

# 数据文件：假设 CSV 和本脚本在同一个文件夹
DATA_PATH = ROOT_DIR / 'ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv'

# 模型保存：直接保存在根目录
MODEL_SAVE_PATH = ROOT_DIR / "xgb_production_pipeline_.pkl"

# ================= 1. 数据加载与清洗 =================
def load_and_clean_raw_data(filepath):
    print(f">>> 正在加载原始数据: {filepath.name}")
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    
    # 筛选 2025 年的数据
    df = df[df['month'].str.startswith('2025')].copy()
    
    # 1.1 算房龄
    df['house_age'] = 2026 - df['lease_commence_date']
    
    # 1.2 算楼层中位数
    def get_median_storey(storey_range_str):
        try:
            parts = str(storey_range_str).split(' TO ')
            if len(parts) == 2:
                return (int(parts[0]) + int(parts[1])) / 2
            return np.nan
        except:
            return np.nan
            
    df['storey_median'] = df['storey_range'].apply(get_median_storey)
    df['log_storey_median'] = np.log1p(df['storey_median'])
    
    # 1.3. 删除无用列
    columns_to_drop = ['month', 'block', 'street_name', 'remaining_lease', 
                       'lease_commence_date', 'storey_range', 'flat_type', 'storey_median']
    df = df.drop(columns=columns_to_drop)
    
    print(f"✅ 数据清洗完成。当前样本量: {df.shape[0]}")
    return df

# ================= 核心主程序 =================
def main():
    print("\n" + "="*40)
    print("🚀 HDB 价格预测流水线启动")
    print("="*40)

    # 检查数据文件是否存在
    if not DATA_PATH.exists():
        print(f"❌ 错误：在当前目录未找到数据文件！")
        print(f"期待路径: {DATA_PATH}")
        return

    # 2.1 加载数据
    df_clean = load_and_clean_raw_data(DATA_PATH) 

    X = df_clean.drop(['resale_price'], axis=1)
    y = df_clean['resale_price']  

    # 2.2 划分数据集
    print(">>> 正在划分数据集...")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # 2.3 组装生产流水线
    print(">>> 正在构建特征工程与 XGBoost 流水线...")
    preprocessing_pipeline = ColumnTransformer(transformers=[
        ('target_enc', TargetEncoder(), ['town', 'flat_model']),
        ('passthrough_nums', 'passthrough', ['floor_area_sqm', 'house_age', 'log_storey_median'])
    ], remainder='drop')

    pipeline = Pipeline([
        ('preprocessor', preprocessing_pipeline),
        ('global_scaler', StandardScaler()),
        ('regressor', XGBRegressor(
            random_state=RANDOM_SEED, 
            objective='reg:squarederror',
            n_estimators=200,      
            max_depth=7,           
            learning_rate=0.1,     
            subsample=0.8,         
            min_child_weight=1     
        )) 
    ])

    # 2.4 训练流水线
    print(">>> 正在训练模型，请稍候...")
    pipeline.fit(x_train, y_train)

    # 3. 流水线预测与评估
    print(">>> 正在进行模型评估...")
    y_pred = pipeline.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"📊 [评估结果] Test R2 Score: {r2:.4f}")

    # 3.2 保存模型到当前根目录
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    
    print(f"\n✅ 成功！模型已保存至根目录: \n👉 {MODEL_SAVE_PATH}")
    print("="*40)
    print(">>> 流程全部结束 <<<\n")

if __name__ == "__main__":
    main()
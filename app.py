import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ================= 1. 页面与路径配置 =================
st.set_page_config(page_title="HDB组屋房价预测系统", page_icon="🏢", layout="wide")

st.title("🏢 新加坡 HDB 组屋转售价格预测系统 (XGBoost版)")
st.markdown("通过左侧面板调整房屋参数，右侧将实时显示预测转售价格。本模型基于2025年最新市场数据训练。")


# ================= 2. 加载模型 =================
@st.cache_resource
def load_model():
    return joblib.load("xgb_production_pipeline.pkl")

try:
    model = load_model()
    st.sidebar.success("✅ 生产级 Pipeline 模型已就绪")
except Exception as e:
    st.sidebar.error(f"❌ 无法加载模型，请检查文件路径: {e}")
    st.stop()

# ================= 3. 侧边栏：用户输入参数 =================
st.sidebar.header("输入房屋特征参数")

def user_input_features():
    # 类别特征下拉菜单 (这里列出几个常见的供演示，你可以根据你的实际数据补充完整)
    towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'CENTRAL AREA', 'CLEMENTI', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SENGKANG', 'TAMPINES', 'WOODLANDS', 'YISHUN']
    flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified', 'Premium Apartment', 'Maisonette', 'Apartment', 'DBSS', 'Type S1']
    
    town = st.sidebar.selectbox("所在市镇 (Town)", towns)
    flat_model = st.sidebar.selectbox("组屋型号 (Flat Model)", flat_models)
    
    # 数值特征滑动条
    floor_area_sqm = st.sidebar.slider("房屋面积 (平方米)", 30.0, 150.0, 90.0)
    house_age = st.sidebar.slider("房屋楼龄 (年)", 0, 99, 10)
    
    # 为了UI友好，让用户输入实际楼层，我们在后台转换为模型需要的 log_storey_median
    storey = st.sidebar.slider("所在楼层", 1, 50, 10)
    log_storey_median = np.log1p(storey) # 应用与训练时相同的对数变换
    
    # 组装成DataFrame，注意列名必须与你训练 Pipeline 时传入的列名完全一致！
    data = {
        'town': town,
        'flat_model': flat_model,
        'floor_area_sqm': floor_area_sqm,
        'house_age': house_age,
        'log_storey_median': log_storey_median
    }
    return pd.DataFrame(data, index=[0]), storey

input_df, display_storey = user_input_features()

# ================= 4. 主界面显示 =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 当前输入参数")
    # 为了展示给用户看，我们把后台的 log_storey_median 换回直观的楼层数
    display_df = input_df.copy()
    display_df = display_df.drop(columns=['log_storey_median'])
    display_df['storey'] = display_storey
    st.write(display_df)

with col2:
    st.subheader("💰 预测结果")
    # 将包含 5 个特征的原始 DataFrame 喂给 Pipeline
    # Pipeline 会自动执行 TargetEncoder 和 StandardScaler，然后进行预测
    prediction = model.predict(input_df)
    final_price = prediction[0]
    
    st.metric(label="预测转售房价 (SGD)", value=f"S$ {final_price:,.2f}")
    
    # 根据组屋价格给出简单评价 (价格阈值可根据2025年实际行情调整)
    if final_price > 800000:
        st.warning("🔥 这是一套高价位/热门房源。")
    elif final_price < 400000:
        st.info("💡 这是一套经济型/高性价比房源。")
    else:
        st.success("✅ 这是一套中等价位房源。")

# ================= 5. 地图展示 (新增部分) =================
st.markdown("---") # 加一条分割线
st.subheader("📍 该市镇大致地理位置")

# 新加坡各市镇的近似中心经纬度字典
town_coordinates = {
    'ANG MO KIO': [1.3691, 103.8454],
    'BEDOK': [1.3236, 103.9273],
    'BISHAN': [1.3526, 103.8484],
    'BUKIT BATOK': [1.3496, 103.7497],
    'BUKIT MERAH': [1.2819, 103.8167],
    'CENTRAL AREA': [1.2789, 103.8536],
    'CHOA CHU KANG': [1.3850, 103.7445],
    'CLEMENTI': [1.3162, 103.7649],
    'GEYLANG': [1.3201, 103.8918],
    'HOUGANG': [1.3713, 103.8925],
    'JURONG EAST': [1.3329, 103.7436],
    'JURONG WEST': [1.3404, 103.7090],
    'KALLANG/WHAMPOA': [1.3100, 103.8651],
    'MARINE PARADE': [1.3020, 103.8971],
    'PASIR RIS': [1.3721, 103.9474],
    'PUNGGOL': [1.3984, 103.9023],
    'QUEENSTOWN': [1.2942, 103.8058],
    'SEMBAWANG': [1.4491, 103.8185],
    'SENGKANG': [1.3868, 103.8914],
    'SERANGOON': [1.3554, 103.8679],
    'TAMPINES': [1.3496, 103.9568],
    'TOA PAYOH': [1.3343, 103.8501],
    'WOODLANDS': [1.4360, 103.7861],
    'YISHUN': [1.4294, 103.8350]
}

# 获取用户当前选择的市镇的经纬度 (如果找不到，默认显示新加坡中心点)
current_town = input_df['town'].iloc[0]
coords = town_coordinates.get(current_town, [1.3521, 103.8198])

# 转换成 Streamlit 地图需要的 DataFrame 格式
map_df = pd.DataFrame({
    'lat': [coords[0]],
    'lon': [coords[1]]
})

# 渲染地图
st.map(map_df, zoom=11)
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Dự Đoán Bệnh Tim", page_icon="❤️", layout="centered")

# Load mô hình và scaler
@st.cache_resource
def load_model():
    return keras.models.load_model("heart_disease_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

mo_hinh = load_model()
scaler = load_scaler()

# Danh sách cột đầu vào
features = ['id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Giao diện người dùng
st.subheader("NHÓM 1: P.Nam, H.Nam, P.Huy, T.Tiến")
st.title("🔍 Ứng dụng Dự Đoán Bệnh Tim")
st.markdown("### **Nhập thông tin để kiểm tra nguy cơ mắc bệnh tim** 🏥")

# Bố cục giao diện
with st.container():
    st.subheader("📋 Nhập thông tin bệnh nhân:")
    age = st.slider("Tuổi", 1, 150, 50)
    sex = st.radio("Giới tính", ["Nam", "Nữ"], horizontal=True)
    cp = st.selectbox("Loại đau ngực", [0, 1, 2, 3])
    trestbps = st.slider("Huyết áp khi nghỉ (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Đường huyết > 120 mg/dl", [0, 1], horizontal=True)
    restecg = st.selectbox("Kết quả ECG khi nghỉ", [0, 1, 2])
    thalch = st.slider("Nhịp tim tối đa đạt được", 70, 220, 150)
    exang = st.radio("Đau thắt ngực khi tập không?", [0, 1], horizontal=True)
    oldpeak = st.slider("Chênh ST do tập thể dục", 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox("Độ dốc của đoạn ST", [0, 1, 2])
    ca = st.slider("Số lượng mạch chính có cản trở (0-4)", 0, 4, 1)
    thal = st.selectbox("Thalassemia", [3, 6, 7])

# Dự đoán kết quả
if st.button("📌 Dự đoán ngay"):
    with st.spinner("⏳ Đang phân tích dữ liệu..."):
        sex_numeric = 1 if sex == "Nam" else 0
        input_data = pd.DataFrame([[0, age, sex_numeric, 0, cp, trestbps, chol, fbs, restecg, 
                                    thalch, exang, oldpeak, slope, ca, thal]], columns=features)
        input_data[['thalch', 'chol', 'trestbps', 'ca']] = scaler.transform(input_data[['thalch', 'chol', 'trestbps', 'ca']])
        prediction = mo_hinh.predict(input_data)
        risk = prediction.flatten()[0]

    # Hiển thị kết quả
    st.markdown("---")
    st.subheader("🔹 Kết quả Dự Đoán")
    if risk > 0.5:
        st.error(f"⚠️ **Nguy cơ mắc bệnh tim: {risk:.2%}**")
    else:
        st.success(f"✅ **Không có nguy cơ mắc bệnh tim: {risk:.2%}**")
    
    # Biểu đồ trực quan
    st.markdown("#### 📊 Phân tích nguy cơ")
    
    # Biểu đồ gauge
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        title={"text": "Nguy cơ mắc bệnh tim (%)"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "red"},
               "steps": [{"range": [0, 50], "color": "grey"},
                          {"range": [50, 100], "color": "grey"}]}
    ))
    st.plotly_chart(fig1, use_container_width=True)
    
    # Biểu đồ cột
    fig2, ax = plt.subplots()
    ax.bar(["An toàn", "Nguy cơ"], [1-risk, risk], color=["green", "black"])
    ax.set_ylabel("Xác suất")
    ax.set_title("So sánh mức độ nguy cơ")
    st.pyplot(fig2)
    
    # Biểu đồ tròn
    labels = ["Không mắc bệnh", "Mắc bệnh"]
    values = [1-risk, risk]
    fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig3.update_traces(marker=dict(colors=["green", "purple"]))
    st.plotly_chart(fig3, use_container_width=True)
    
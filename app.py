import streamlit as st
import pandas as pd
import os
from utils.fetcher import fetch_latest_data, load_data
from models.stats_model import StatsModel
from models.gnn_model import GNNModel

st.set_page_config(page_title="6-Strike-Bomb 로또 예측기", page_icon="🎱")

st.title("🎱 6-Strike-Bomb 로또 6/45 예측기")
st.markdown("### 인공지능 기반 로또 번호 예측 시스템")

# Sidebar
st.sidebar.header("설정 (Settings)")

if st.sidebar.button("데이터 업데이트 (Update Data)"):
    with st.spinner("데이터를 가져오는 중입니다... (Fetching Data...)"):
        df = fetch_latest_data()
        st.sidebar.success(f"데이터 업데이트 완료! 총 {len(df)} 회차")

# Load Data
df = load_data()
if df.empty:
    st.warning("데이터가 없습니다. 사이드바에서 '데이터 업데이트'를 눌러주세요.")
    st.stop()

st.info(f"현재 데이터: **{len(df)}** 회차까지 저장되어 있습니다.")

# Model Selection
model_name = st.selectbox(
    "알고리즘 선택 (Select Algorithm)",
    ["통계 기반 (Stats Based)", "GNN (Graph Neural Network)"]
)

# Generate Button
if st.button("예측 번호 생성 (Generate Prediction)"):
    with st.spinner("모델 학습 및 예측 중... (Thinking...)"):
        try:
            model = None
            if "Stats" in model_name:
                model = StatsModel()
            elif "GNN" in model_name:
                model = GNNModel()
            
            # Train on the spot (fast enough for this scale)
            model.train(df)
            prediction = model.predict()
            
            st.success("예측된 번호 (Predicted Numbers):")
            
            # Display nicely
            cols = st.columns(6)
            for i, num in enumerate(prediction):
                cols[i].metric(label=f"Num {i+1}", value=num)
                
            st.balloons()
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            # For debugging
            import traceback
            st.text(traceback.format_exc())

st.markdown("---")
st.markdown("#### 최근 당첨 번호 (Latest Winning Numbers)")
st.dataframe(df.sort_values(by='drwNo', ascending=False).head(5)[['drwNo', 'drwNoDate', 'drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo']])

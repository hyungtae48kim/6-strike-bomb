import streamlit as st
import pandas as pd
import os
from utils.fetcher import fetch_latest_data, load_data, add_manual_data
from utils.history_manager import HistoryManager
from models.stats_model import StatsModel
from models.gnn_model import GNNModel
from models.bayes_model import BayesModel
from models.weighted_ensemble_model import WeightedEnsembleModel
from models.enums import AlgorithmType

st.set_page_config(page_title="6-Strike-Bomb 로또 예측기", page_icon="🎱")

st.title("🎱 6-Strike-Bomb 로또 6/45 예측기")
st.markdown("### 인공지능 기반 로또 번호 예측 시스템")

# Initialize History Manager
history_manager = HistoryManager()

# Load Data
df = load_data()


# Sidebar
st.sidebar.header("설정 (Settings)")

if st.sidebar.button("데이터 업데이트 (Update Data)"):
    with st.spinner("데이터를 가져오는 중입니다... (Fetching Data...)"):
        df, message = fetch_latest_data()
        # Also update hit counts processing
        history_manager.update_hit_counts(df)
        if "성공" in message:
            st.sidebar.success(message)
        else:
            st.sidebar.warning(message)

# Manual Update Section
with st.sidebar.expander("수동 입력 (Manual Update)"):
    st.markdown("API 오류 시 직접 입력하세요.")
    
    # Calculate next draw number and date
    last_draw = 1
    last_date = None
    if not df.empty:
        last_draw = int(df['drwNo'].max())
        last_date = pd.to_datetime(df[df['drwNo'] == last_draw]['drwNoDate'].values[0])
    
    next_draw = last_draw + 1
    next_date = (last_date + pd.Timedelta(days=7)).date() if last_date else datetime.today().date()
    
    m_drwNo = st.number_input("회차 (Draw No)", min_value=1, value=next_draw, step=1)
    m_date = st.date_input("날짜 (Date)", value=next_date)
    
    st.markdown("당첨 번호 (Winning Numbers)")
    c1, c2, c3 = st.columns(3)
    n1 = c1.number_input("No 1", min_value=1, max_value=45, key="n1")
    n2 = c2.number_input("No 2", min_value=1, max_value=45, key="n2")
    n3 = c3.number_input("No 3", min_value=1, max_value=45, key="n3")
    
    c4, c5, c6 = st.columns(3)
    n4 = c4.number_input("No 4", min_value=1, max_value=45, key="n4")
    n5 = c5.number_input("No 5", min_value=1, max_value=45, key="n5")
    n6 = c6.number_input("No 6", min_value=1, max_value=45, key="n6")
    
    bonus = st.number_input("보너스 (Bonus)", min_value=1, max_value=45, key="bn")
    
    if st.button("저장 (Save)"):
        nums = [n1, n2, n3, n4, n5, n6]
        if len(set(nums)) != 6:
            st.error("중복된 번호가 있습니다!")
        elif bonus in nums:
             st.error("보너스 번호가 당첨 번호와 겹칩니다!")
        else:
            success, msg = add_manual_data(m_drwNo, m_date.strftime("%Y-%m-%d"), sorted(nums), bonus)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

# Data Check
if df.empty:
    st.warning("데이터가 없습니다. 사이드바에서 '데이터 업데이트'를 눌러주세요.")
    st.stop()

# Update hit counts on startup (just in case)
history_manager.update_hit_counts(df)

st.info(f"현재 데이터: **{len(df)}** 회차까지 저장되어 있습니다.")

# Display Algorithm Weights
st.sidebar.markdown("---")
st.sidebar.header("알고리즘 가중치 (Weights)")
weights = history_manager.get_weights()
for alg, weight in weights.items():
    st.sidebar.markdown(f"**{alg}**: {weight:.2f}")

# Model Selection
# Create mapping from display name to Enum
alg_map = {alg.value: alg for alg in AlgorithmType}

selected_alg_name = st.selectbox(
    "알고리즘 선택 (Select Algorithm)",
    list(alg_map.keys())
)
selected_alg_enum = alg_map[selected_alg_name]

# Generate Button
if st.button("예측 번호 생성 (Generate Prediction)"):
    with st.spinner("모델 학습 및 예측 중... (Thinking...)"):
        try:
            model = None
            if selected_alg_enum == AlgorithmType.STATS:
                model = StatsModel()
            elif selected_alg_enum == AlgorithmType.GNN:
                model = GNNModel()
            elif selected_alg_enum == AlgorithmType.BAYES:
                model = BayesModel()
            elif selected_alg_enum == AlgorithmType.ENSEMBLE:
                model = WeightedEnsembleModel(weights)
            
            # Train on the spot
            model.train(df)
            prediction = model.predict()
            
            # Save prediction
            # Use next draw number (current max + 1)
            next_draw_no = int(df['drwNo'].max()) + 1
            history_manager.save_prediction(next_draw_no, selected_alg_enum, prediction)

            st.success(f"예측된 번호 (Predicted Numbers) - {selected_alg_name}:")
            
            # Display nicely
            cols = st.columns(6)
            for i, num in enumerate(prediction):
                cols[i].metric(label=f"Num {i+1}", value=num)
                
            st.balloons()
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            import traceback
            st.text(traceback.format_exc())

st.markdown("---")
st.markdown("#### 최근 당첨 번호 (Latest Winning Numbers)")
st.dataframe(df.sort_values(by='drwNo', ascending=False).head(5)[['drwNo', 'drwNoDate', 'drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo']])

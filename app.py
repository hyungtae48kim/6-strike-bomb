import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils.fetcher import fetch_latest_data, load_data, add_manual_data
from utils.history_manager import HistoryManager
from models.stats_model import StatsModel
from models.gnn_model import GNNModel
from models.bayes_model import BayesModel
from models.weighted_ensemble_model import WeightedEnsembleModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.deepsets_model import DeepSetsModel
from models.pagerank_model import PageRankModel
from models.community_model import CommunityModel
from models.markov_model import MarkovModel
from models.pattern_model import PatternModel
from models.montecarlo_model import MonteCarloModel
from models.ultimate_ensemble_model import UltimateEnsembleModel
from models.stacking_ensemble_model import StackingEnsembleModel
from models.enums import AlgorithmType
from utils.analysis import LottoAnalyzer, CombinationFilter
from utils.wheeling import WheelingSystem

st.set_page_config(page_title="6-Strike-Bomb 로또 예측기", page_icon="🎱", layout="wide")

# 한글 폰트 및 이모지 설정 (Linux 환경 깨짐 방지)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', '맑은 고딕', 'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🎱 6-Strike-Bomb 로또 6/45 예측기")
st.markdown("### Ultimate AI 기반 로또 번호 예측 시스템")
st.markdown("*11개 AI 모델을 통합한 메타 앙상블 시스템 + 조합 최적화*")

# Initialize History Manager
history_manager = HistoryManager()

# Load Data
df = load_data()

# Analyzer 초기화
analyzer = LottoAnalyzer()


# Model Factory
def create_model(alg_type: AlgorithmType, weights: dict = None):
    """알고리즘 타입에 따른 모델 생성"""
    model_map = {
        AlgorithmType.STATS: lambda: StatsModel(),
        AlgorithmType.GNN: lambda: GNNModel(),
        AlgorithmType.BAYES: lambda: BayesModel(),
        AlgorithmType.ENSEMBLE: lambda: WeightedEnsembleModel(weights or {}),
        AlgorithmType.LSTM: lambda: LSTMModel(epochs=50),
        AlgorithmType.TRANSFORMER: lambda: TransformerModel(epochs=50),
        AlgorithmType.DEEPSETS: lambda: DeepSetsModel(epochs=50),
        AlgorithmType.PAGERANK: lambda: PageRankModel(),
        AlgorithmType.COMMUNITY: lambda: CommunityModel(),
        AlgorithmType.MARKOV: lambda: MarkovModel(),
        AlgorithmType.PATTERN: lambda: PatternModel(),
        AlgorithmType.MONTECARLO: lambda: MonteCarloModel(n_simulations=5000),
        AlgorithmType.ULTIMATE: lambda: UltimateEnsembleModel(weights or {}),
        AlgorithmType.STACKING: lambda: StackingEnsembleModel(),
    }

    factory = model_map.get(alg_type)
    if factory:
        return factory()
    return None


# Sidebar
st.sidebar.header("⚙️ 설정 (Settings)")

if st.sidebar.button("📥 데이터 업데이트 (Update Data)"):
    with st.spinner("데이터를 가져오는 중입니다... (Fetching Data...)"):
        df, message = fetch_latest_data()
        history_manager.update_hit_counts(df)
        if "성공" in message:
            st.sidebar.success(message)
        else:
            st.sidebar.warning(message)

# Manual Update Section
with st.sidebar.expander("📝 수동 입력 (Manual Update)"):
    st.markdown("API 오류 시 직접 입력하세요.")

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

# Update hit counts on startup
history_manager.update_hit_counts(df)

st.info(f"📊 현재 데이터: **{len(df)}** 회차까지 저장되어 있습니다.")

# Display Algorithm Weights (Advanced)
st.sidebar.markdown("---")
st.sidebar.header("📈 알고리즘 가중치")

# Use advanced weights
weights = history_manager.get_advanced_weights()

# Group algorithms by tier
tier1 = ["Stats Based", "GNN", "Bayes Theorem"]
tier2 = ["LSTM", "Transformer", "DeepSets"]
tier3 = ["PageRank", "Community"]
tier4 = ["Markov Chain", "Pattern Analysis", "Monte Carlo"]

with st.sidebar.expander("Tier 1: 기존 모델", expanded=False):
    for alg in tier1:
        if alg in weights:
            st.markdown(f"**{alg}**: {weights[alg]:.2f}")

with st.sidebar.expander("Tier 2: 딥러닝", expanded=False):
    for alg in tier2:
        if alg in weights:
            st.markdown(f"**{alg}**: {weights[alg]:.2f}")

with st.sidebar.expander("Tier 3: 그래프", expanded=False):
    for alg in tier3:
        if alg in weights:
            st.markdown(f"**{alg}**: {weights[alg]:.2f}")

with st.sidebar.expander("Tier 4: 확률/패턴", expanded=False):
    for alg in tier4:
        if alg in weights:
            st.markdown(f"**{alg}**: {weights[alg]:.2f}")

# Model Selection with categories
st.markdown("---")
st.subheader("🎯 알고리즘 선택")

# Algorithm categories
algorithm_categories = {
    "🏆 Ultimate (권장)": [AlgorithmType.ULTIMATE, AlgorithmType.STACKING],
    "📊 기존 모델": [AlgorithmType.STATS, AlgorithmType.BAYES, AlgorithmType.GNN, AlgorithmType.ENSEMBLE],
    "🧠 딥러닝": [AlgorithmType.LSTM, AlgorithmType.TRANSFORMER, AlgorithmType.DEEPSETS],
    "🔗 그래프": [AlgorithmType.PAGERANK, AlgorithmType.COMMUNITY],
    "🎲 확률/패턴": [AlgorithmType.MARKOV, AlgorithmType.PATTERN, AlgorithmType.MONTECARLO],
}

col1, col2 = st.columns([1, 2])

with col1:
    category = st.selectbox("카테고리 선택", list(algorithm_categories.keys()))

with col2:
    available_algs = algorithm_categories[category]
    alg_map = {alg.value: alg for alg in available_algs}
    selected_alg_name = st.selectbox("알고리즘 선택", list(alg_map.keys()))
    selected_alg_enum = alg_map[selected_alg_name]

# Algorithm description
alg_descriptions = {
    AlgorithmType.ULTIMATE: "11개 모델 통합 메타 앙상블 + 조건부 확률 샘플링 + 조합 필터링",
    AlgorithmType.STACKING: "7개 모델의 확률 분포를 메타 모델(Ridge)로 학습하는 스태킹 앙상블",
    AlgorithmType.STATS: "Z-score와 softmax 가중치를 사용한 통계 분석",
    AlgorithmType.BAYES: "Beta-Binomial 켤레 사전분포 기반 베이즈 추론",
    AlgorithmType.GNN: "번호 동시출현 그래프를 GCN으로 분석",
    AlgorithmType.ENSEMBLE: "기존 3개 모델의 가중 앙상블",
    AlgorithmType.LSTM: "LSTM 시계열 딥러닝 모델 (BCEWithLogitsLoss)",
    AlgorithmType.TRANSFORMER: "Self-Attention 기반 Transformer 모델 (BCEWithLogitsLoss)",
    AlgorithmType.DEEPSETS: "순서 불변 DeepSets + GRU 시간적 모델 (집합 인코딩)",
    AlgorithmType.PAGERANK: "동시출현 그래프에서 PageRank 중심성 분석",
    AlgorithmType.COMMUNITY: "Louvain 알고리즘으로 번호 클러스터 탐지",
    AlgorithmType.MARKOV: "번호 전이 확률 기반 마르코프 체인",
    AlgorithmType.PATTERN: "주기성, 간격, 합계 패턴 종합 분석",
    AlgorithmType.MONTECARLO: "몬테카를로 시뮬레이션으로 최적 조합 탐색",
}

st.caption(f"ℹ️ {alg_descriptions.get(selected_alg_enum, '')}")

# Generate Button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    n_games = st.number_input("🎮 생성할 게임 수 (Number of Games)", min_value=1, max_value=10, value=1)

with col2:
    generate_clicked = st.button("🚀 예측 번호 생성", type="primary", use_container_width=True)

if generate_clicked:
    try:
        model = create_model(selected_alg_enum, weights)

        if model is None:
            st.error("알 수 없는 알고리즘입니다.")
        else:
            # Progress bar 및 상태 텍스트
            progress_bar = st.progress(0, text="🧠 모델 학습 중... (Training...)")

            # Train model
            model.train(df)
            progress_bar.progress(0.3, text="🧠 모델 학습 완료. 예측 생성 중...")

            # 예측 생성
            next_draw_no = int(df['drwNo'].max()) + 1

            if n_games > 1 and hasattr(model, 'predict_multiple'):
                # predict_multiple 지원 모델 (Ultimate, Stacking)
                progress_bar.progress(0.5, text="🎯 예측 생성 중...")
                predictions = model.predict_multiple(n_games)
                progress_bar.progress(1.0, text=f"🎯 {n_games}개 게임 생성 완료!")
            else:
                predictions = []
                for i in range(n_games):
                    predictions.append(model.predict())
                    progress = 0.3 + 0.7 * (i + 1) / n_games
                    progress_bar.progress(progress, text=f"🎯 게임 {i + 1}/{n_games} 생성 완료")

            progress_bar.empty()

            game_suffix = f" ({n_games}게임)" if n_games > 1 else ""
            st.success(f"✨ 예측된 번호 (Predicted Numbers) - {selected_alg_name}{game_suffix}:")

            # 각 게임별 결과 표시
            for game_idx, prediction in enumerate(predictions):
                # Save prediction
                history_manager.save_prediction(next_draw_no, selected_alg_enum, prediction)

                if n_games > 1:
                    st.markdown(f"#### 🎮 Game {game_idx + 1}")

                # Display nicely with colored balls
                cols = st.columns(6)
                for i, num in enumerate(prediction):
                    with cols[i]:
                        # Color based on number range
                        if num <= 10:
                            color = "#FBC400"  # Yellow
                        elif num <= 20:
                            color = "#69C8F2"  # Blue
                        elif num <= 30:
                            color = "#FF7272"  # Red
                        elif num <= 40:
                            color = "#AAAAAA"  # Gray
                        else:
                            color = "#B0D840"  # Green

                        st.markdown(
                            f"""
                            <div style="
                                background-color: {color};
                                color: white;
                                font-size: 24px;
                                font-weight: bold;
                                border-radius: 50%;
                                width: 60px;
                                height: 60px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                margin: auto;
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                            ">{num}</div>
                            """,
                            unsafe_allow_html=True
                        )

                # 조합 분석 표시
                ac = analyzer.ac_value(prediction)
                s = analyzer.sum_value(prediction)
                odd, even = analyzer.odd_even_ratio(prediction)
                low, high = analyzer.high_low_ratio(prediction)
                consec = analyzer.consecutive_count(prediction)
                ld = analyzer.last_digit_distribution(prediction)
                decades = analyzer.decade_distribution(prediction)
                score = analyzer.comprehensive_score(prediction, df)

                with st.expander(f"📐 조합 분석 (Game {game_idx + 1})" if n_games > 1 else "📐 조합 분석 (Combination Analysis)", expanded=(n_games == 1)):
                    a1, a2, a3, a4 = st.columns(4)
                    with a1:
                        st.metric("AC값", ac, help="7-10이 이상적 (역대 당첨번호 기준)")
                    with a2:
                        st.metric("합계", s, help="100-170이 이상적 (역대 평균 ~135)")
                    with a3:
                        st.metric("홀:짝", f"{odd}:{even}", help="2:4~4:2가 이상적")
                    with a4:
                        st.metric("종합 점수", f"{score:.0f}", help="높을수록 역대 패턴에 부합")

                    a5, a6, a7 = st.columns(3)
                    with a5:
                        st.metric("저:고", f"{low}:{high}", help="1-22:23-45 비율")
                    with a6:
                        st.metric("연번 쌍", consec, help="0-2가 이상적")
                    with a7:
                        decade_str = " / ".join(f"{k}:{v}" for k, v in decades.items() if v > 0)
                        st.metric("번호대", decade_str)

            st.balloons()

            # Show top numbers for Ultimate/Stacking model
            has_top = hasattr(model, 'get_top_numbers')
            if has_top and selected_alg_enum in [AlgorithmType.ULTIMATE, AlgorithmType.STACKING]:
                st.markdown("---")
                st.subheader("📊 상위 확률 번호 (Top 10)")
                top_nums = model.get_top_numbers(10)

                top_df = pd.DataFrame(top_nums, columns=["번호", "확률"])
                top_df["확률"] = top_df["확률"].apply(lambda x: f"{x*100:.2f}%")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(top_df, hide_index=True)

            # 휠링 시스템 (Ultimate/Stacking 모델에서 사용 가능)
            if has_top and selected_alg_enum in [AlgorithmType.ULTIMATE, AlgorithmType.STACKING]:
                st.markdown("---")
                st.subheader("🎡 휠링 시스템 (Wheeling System)")
                st.caption("상위 번호에서 수학적 커버리지를 보장하는 조합 세트를 생성합니다.")

                w1, w2 = st.columns(2)
                with w1:
                    n_candidates = st.slider("후보 번호 수", 8, 15, 10,
                                             help="상위 N개 번호를 후보로 선정")
                with w2:
                    guarantee = st.selectbox("보장 등급",
                                             [3, 4, 5],
                                             format_func=lambda x: {3: "5등 보장 (3개 적중)", 4: "4등 보장 (4개 적중)", 5: "3등 보장 (5개 적중)"}[x])

                if st.button("🎡 휠 생성", key="wheel_btn"):
                    top_numbers = [t[0] for t in model.get_top_numbers(n_candidates)]

                    try:
                        ws = WheelingSystem(top_numbers, guarantee_match=guarantee)
                        wheel = ws.generate_abbreviated_wheel()
                        report = ws.get_coverage_report(wheel)

                        st.success(f"✅ {report['총_티켓_수']}장의 티켓 생성 (완전 휠 대비 {report['절감_비율']} 절감)")
                        st.markdown(f"**후보 번호**: {report['후보_번호']}")
                        st.markdown(f"**보장**: {report['보장_등수']} ({report['커버리지']} 커버리지)")

                        # 티켓 목록 표시
                        ticket_data = []
                        for idx, ticket in enumerate(wheel):
                            ticket_data.append({
                                "티켓 #": idx + 1,
                                "번호": ", ".join(map(str, ticket)),
                                "합계": sum(ticket),
                                "AC값": analyzer.ac_value(ticket)
                            })
                        st.dataframe(pd.DataFrame(ticket_data), hide_index=True, use_container_width=True)
                    except Exception as e:
                        st.error(f"휠링 오류: {e}")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        import traceback
        with st.expander("상세 오류 정보"):
            st.text(traceback.format_exc())

# Latest winning numbers
st.markdown("---")
st.subheader("📋 최근 당첨 번호 (Latest Winning Numbers)")
st.dataframe(
    df.sort_values(by='drwNo', ascending=False).head(10)[
        ['drwNo', 'drwNoDate', 'drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo']
    ],
    hide_index=True,
    use_container_width=True
)

# Algorithm Performance Stats
st.markdown("---")
st.subheader("📈 알고리즘 성능 통계")

stats = history_manager.get_algorithm_stats()
if stats:
    stats_df = pd.DataFrame([
        {
            "알고리즘": alg,
            "예측 수": data['count'],
            "평균 적중": f"{data['avg']:.2f}",
            "최대 적중": data['max'],
            "표준편차": f"{data['std']:.2f}"
        }
        for alg, data in stats.items() if data['count'] > 0
    ])

    if not stats_df.empty:
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    else:
        st.info("아직 검증된 예측 기록이 없습니다. 예측 후 당첨 결과가 나오면 자동으로 업데이트됩니다.")

# 모델 검증 섹션
st.markdown("---")
with st.expander("🔬 모델 검증 (Walk-Forward Validation)", expanded=False):
    st.caption("Walk-Forward 시간적 검증으로 모델의 실제 예측력과 과적합 여부를 측정합니다.")

    val_model_options = {
        "Stats Based": {"class": StatsModel, "kwargs": {}},
        "Bayes Theorem": {"class": BayesModel, "kwargs": {}},
        "Markov Chain": {"class": MarkovModel, "kwargs": {}},
        "PageRank": {"class": PageRankModel, "kwargs": {}},
        "Pattern Analysis": {"class": PatternModel, "kwargs": {}},
    }

    val_model_name = st.selectbox("검증할 모델", list(val_model_options.keys()), key="val_model")

    if st.button("🔬 검증 실행", key="val_btn"):
        with st.spinner(f"{val_model_name} 검증 중... (수 분 소요)"):
            try:
                from utils.validation import WalkForwardValidator

                validator = WalkForwardValidator(
                    initial_train_size=500,
                    test_size=5,
                    step_size=100
                )

                config = val_model_options[val_model_name]
                result = validator.validate(config["class"], df, **config["kwargs"])

                st.success(f"검증 완료: {result.n_folds}개 폴드")

                v1, v2, v3, v4 = st.columns(4)
                with v1:
                    st.metric("평균 적중", f"{result.avg_hits:.3f}")
                with v2:
                    st.metric("표준편차", f"{result.std_hits:.3f}")
                with v3:
                    st.metric("과적합 갭", f"{result.overfit_gap:.3f}",
                              help="학습-테스트 적중 차이 (0.3 이상이면 과적합 의심)")
                with v4:
                    overfit = WalkForwardValidator.detect_overfit(result)
                    st.metric("과적합 진단", overfit["severity"])

                # 적중 분포 표시
                if result.hit_distribution:
                    dist_df = pd.DataFrame([
                        {"적중 수": k, "횟수": v}
                        for k, v in sorted(result.hit_distribution.items())
                    ])
                    st.bar_chart(dist_df.set_index("적중 수"))

            except Exception as e:
                st.error(f"검증 오류: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    ⚠️ 이 소프트웨어는 교육 및 엔터테인먼트 목적으로만 사용해야 합니다.<br>
    로또 번호는 무작위이며, 이 소프트웨어는 당첨을 보장하지 않습니다.
</div>
""", unsafe_allow_html=True)

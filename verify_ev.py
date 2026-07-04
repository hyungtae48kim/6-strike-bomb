"""EV 휴리스틱 검증.

각 회차 당첨조합의 인기도 factor 점수로 실제 1등 당첨자 수를 회귀한다.
양(+)의 유의한 계수 = 휴리스틱이 실제 인기도를 포착함을 반증 가능하게
입증. 같은 회귀로 factor 가중치를 보정한다(하이브리드 접근).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from utils.popularity_model import PopularityModel, DEFAULT_WEIGHTS
from utils.fetcher import load_data
from utils.winner_data import load_winner_data

FACTOR_ORDER = list(DEFAULT_WEIGHTS.keys())


def build_dataset(lotto_df: pd.DataFrame, winner_df: pd.DataFrame,
                  recent_window: int = 10):
    """각 회차 당첨조합 → factor 점수 X, 1등 당첨자수 y, 판매액 sales.

    recent_reuse factor는 시간 누수를 막기 위해 해당 회차의 '직전' 회차들만
    참조해 계산한다(그 회차 추첨 시점에 사람들이 실제로 본 정보와 동일).
    이 때문에 회차마다 직전 이력으로 PopularityModel을 새로 구성한다.
    """
    merged = lotto_df.merge(winner_df, on="drwNo", how="inner").sort_values("drwNo")
    cols = [f"drwtNo{i}" for i in range(1, 7)]
    X, y, sales = [], [], []
    for _, row in merged.iterrows():
        drw = int(row["drwNo"])
        prior = lotto_df[lotto_df["drwNo"] < drw]  # 직전 회차만(누수 방지)
        pm = PopularityModel(df=prior, recent_window=recent_window)
        combo = [int(row[c]) for c in cols]
        fs = pm.factor_scores(combo)
        X.append([fs[f] for f in FACTOR_ORDER])
        y.append(float(row["firstPrzwnerCo"]))
        sales.append(float(row["totSellamnt"]))
    return np.array(X), np.array(y), np.array(sales)


def run_regression(X: np.ndarray, y: np.ndarray, sales: np.ndarray) -> dict:
    """factor 점수로 1등 당첨자 수 회귀. log(판매액) 통제.

    r2_in_sample은 학습 데이터에 그대로 채점한 값이라 예측력이 아닌 적합도
    참고용이다. 실제 예측력은 표본이 충분할 때(n>=10) 교차검증 R²(cv_r2)로
    보고한다.
    """
    log_sales = np.log(np.clip(sales, 1, None)).reshape(-1, 1)
    features = np.hstack([X, log_sales])
    model = Ridge(alpha=1.0)
    model.fit(features, y)
    r2_in_sample = float(model.score(features, y))
    cv_r2 = None
    n = len(y)
    if n >= 10:
        folds = min(5, n)
        scores = cross_val_score(Ridge(alpha=1.0), features, y,
                                 cv=folds, scoring="r2")
        cv_r2 = float(np.mean(scores))
    factor_coefs = {f: float(c) for f, c in
                    zip(FACTOR_ORDER, model.coef_[:len(FACTOR_ORDER)])}
    return {"r2_in_sample": r2_in_sample, "cv_r2": cv_r2,
            "factor_coefs": factor_coefs,
            "sales_coef": float(model.coef_[-1])}


def calibrate_weights(factor_coefs: dict) -> dict:
    """회귀 계수(양수=인기 기여)를 정규화해 가중치로 변환.

    모든 계수가 비양수(검증 실패)면 기본 가중치를 유지한다.
    """
    pos = {f: max(c, 0.0) for f, c in factor_coefs.items()}
    total = sum(pos.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {f: v / total for f, v in pos.items()}


def main():
    lotto_df = load_data()
    winner_df = load_winner_data()
    if winner_df.empty:
        print("검증 데이터 없음. 먼저 수집하세요:")
        print("  python -c \"from utils.winner_data import fetch_winner_data as f; f(1, 1230)\"")
        return
    X, y, sales = build_dataset(lotto_df, winner_df)
    if len(y) < 30:
        print(f"검증 표본 부족(n={len(y)}). 기본 가중치를 유지합니다.")
        return
    result = run_regression(X, y, sales)
    print(f"검증 표본 수: {len(y)}회차")
    print(f"R² (in-sample, 적합도 참고용) = {result['r2_in_sample']:.4f}")
    if result["cv_r2"] is not None:
        print(f"R² (교차검증, 실제 예측력)   = {result['cv_r2']:.4f}")
    else:
        print("R² (교차검증) = 표본 부족으로 생략")
    print("factor별 계수(양수=인기도↑ 기여):")
    for f, c in result["factor_coefs"].items():
        print(f"  {f:16s} {c:+.4f}")
    weights = calibrate_weights(result["factor_coefs"])
    print("보정된 가중치:", {k: round(v, 3) for k, v in weights.items()})


if __name__ == "__main__":
    main()

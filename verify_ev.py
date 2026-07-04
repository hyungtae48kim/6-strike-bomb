"""EV 휴리스틱 검증.

각 회차 당첨조합의 인기도 factor 점수로 실제 1등 당첨자 수를 회귀한다.
양(+)의 유의한 계수 = 휴리스틱이 실제 인기도를 포착함을 반증 가능하게
입증. 같은 회귀로 factor 가중치를 보정한다(하이브리드 접근).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from utils.popularity_model import PopularityModel, DEFAULT_WEIGHTS
from utils.fetcher import load_data
from utils.winner_data import load_winner_data

FACTOR_ORDER = list(DEFAULT_WEIGHTS.keys())


def build_dataset(lotto_df: pd.DataFrame, winner_df: pd.DataFrame):
    """각 회차 당첨조합 → factor 점수 X, 1등 당첨자수 y, 판매액 sales."""
    pm = PopularityModel()
    merged = lotto_df.merge(winner_df, on="drwNo", how="inner")
    cols = [f"drwtNo{i}" for i in range(1, 7)]
    X, y, sales = [], [], []
    for _, row in merged.iterrows():
        combo = [int(row[c]) for c in cols]
        fs = pm.factor_scores(combo)
        X.append([fs[f] for f in FACTOR_ORDER])
        y.append(float(row["firstPrzwnerCo"]))
        sales.append(float(row["totSellamnt"]))
    return np.array(X), np.array(y), np.array(sales)


def run_regression(X: np.ndarray, y: np.ndarray, sales: np.ndarray) -> dict:
    """factor 점수로 1등 당첨자 수 회귀. log(판매액) 통제."""
    log_sales = np.log(np.clip(sales, 1, None)).reshape(-1, 1)
    features = np.hstack([X, log_sales])
    model = Ridge(alpha=1.0)
    model.fit(features, y)
    r2 = float(model.score(features, y))
    factor_coefs = {f: float(c) for f, c in
                    zip(FACTOR_ORDER, model.coef_[:len(FACTOR_ORDER)])}
    return {"r2": r2, "factor_coefs": factor_coefs,
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
    print(f"R² = {result['r2']:.4f}")
    print("factor별 계수(양수=인기도↑ 기여):")
    for f, c in result["factor_coefs"].items():
        print(f"  {f:16s} {c:+.4f}")
    weights = calibrate_weights(result["factor_coefs"])
    print("보정된 가중치:", {k: round(v, 3) for k, v in weights.items()})


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict
from models.enums import AlgorithmType

HISTORY_FILE = 'data/prediction_history.csv'


class HistoryManager:
    """
    예측 히스토리 및 알고리즘 가중치 관리 클래스.
    과거 예측 결과를 저장하고, 성능 기반 동적 가중치를 계산합니다.
    """

    def __init__(self):
        self._ensure_history_file()

    def _ensure_history_file(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        
        if not os.path.exists(HISTORY_FILE):
            df = pd.DataFrame(columns=[
                'draw_no', 'algorithm', 'predicted_numbers', 'hit_count', 'timestamp'
            ])
            df.to_csv(HISTORY_FILE, index=False)

    def save_prediction(self, draw_no: int, algorithm: AlgorithmType, numbers: List[int]):
        """
        Save a prediction to the history file.
        initial hit_count is -1 (unknown).
        """
        # Convert list to string "1,2,3,4,5,6"
        numbers_str = ",".join(map(str, sorted(numbers)))
        
        new_row = {
            'draw_no': draw_no,
            'algorithm': algorithm.value,
            'predicted_numbers': numbers_str,
            'hit_count': -1,
            'timestamp': datetime.now().isoformat()
        }
        
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(HISTORY_FILE, index=False)

    def update_hit_counts(self, lotto_df: pd.DataFrame):
        """
        Update hit_count for past predictions based on actual results.
        lotto_df should have 'drwNo', 'drwtNo1'...'drwtNo6'.
        """
        if not os.path.exists(HISTORY_FILE):
            return

        hist_df = pd.read_csv(HISTORY_FILE)
        
        # Filter rows where hit_count is still -1 (unverified)
        pending_mask = hist_df['hit_count'] == -1
        if not pending_mask.any():
            return

        updated = False
        
        # Create a lookup for draw results
        # { draw_no: {1, 2, 3, 4, 5, 6} }
        results_map = {}
        for _, row in lotto_df.iterrows():
            draw_no = int(row['drwNo'])
            winning_numbers = {
                int(row[f'drwtNo{i}']) for i in range(1, 7)
            }
            results_map[draw_no] = winning_numbers

        # Iterate only pending rows
        for idx in hist_df[pending_mask].index:
            draw_no = int(hist_df.at[idx, 'draw_no'])
            
            if draw_no in results_map:
                pred_str = str(hist_df.at[idx, 'predicted_numbers'])
                pred_nums = set(map(int, pred_str.split(',')))
                
                winning_nums = results_map[draw_no]
                hits = len(pred_nums.intersection(winning_nums))
                
                hist_df.at[idx, 'hit_count'] = hits
                updated = True
        
        if updated:
            hist_df.to_csv(HISTORY_FILE, index=False)
            print("Updated hit counts for past predictions.")

    def get_weights(self) -> Dict[str, float]:
        """
        Calculate weights for each algorithm based on average hit_count.
        Base weight is 1.0.
        Weight = 1.0 + (Average Hit Count / 6.0) * Scale Factor?
        Or simply Average Hit Count (if 0, then epsilon).
        
        Let's try a simple approach:
        Average Hit Count. If no history, default to 1.0.
        """
        if not os.path.exists(HISTORY_FILE):
            return {alg.value: 1.0 for alg in AlgorithmType if alg != AlgorithmType.ENSEMBLE}

        df = pd.read_csv(HISTORY_FILE)
        
        # Filter checked rows
        checked_df = df[df['hit_count'] != -1]
        
        weights = {}
        all_algs = [alg.value for alg in AlgorithmType if alg != AlgorithmType.ENSEMBLE]
        
        if checked_df.empty:
            return {alg: 1.0 for alg in all_algs}

        # Calculate average hit count per algorithm
        avg_hits = checked_df.groupby('algorithm')['hit_count'].mean()
        
        for alg in all_algs:
            # Default to 1.0 if no data
            val = avg_hits.get(alg, 1.0)
            # Ensure it's not 0 to avoid completely ignoring it?
            # Or strict feedback: if it sucks, it gets low weight.
            # Let's give a small epsilon floor.
            weights[alg] = max(float(val), 0.1)

        return weights

    def get_advanced_weights(self, decay_factor: float = 0.9,
                             recent_n: int = 20) -> Dict[str, float]:
        """
        고급 가중치 계산 시스템.

        특징:
        - 최근 N회차 성능에 더 높은 가중치 (지수 감쇠)
        - 성능 변화 트렌드 반영
        - 다양성 보너스

        Args:
            decay_factor: 지수 감쇠 계수 (0.9 = 최근 10%씩 더 중요)
            recent_n: 최근 고려할 예측 개수

        Returns:
            각 알고리즘의 가중치 딕셔너리
        """
        excluded_algs = [AlgorithmType.ENSEMBLE, AlgorithmType.ULTIMATE]
        all_algs = [alg.value for alg in AlgorithmType if alg not in excluded_algs]

        if not os.path.exists(HISTORY_FILE):
            return {alg: 1.0 for alg in all_algs}

        df = pd.read_csv(HISTORY_FILE)
        checked_df = df[df['hit_count'] != -1].copy()

        if checked_df.empty:
            return {alg: 1.0 for alg in all_algs}

        # 타임스탬프 기준 정렬
        if 'timestamp' in checked_df.columns:
            checked_df = checked_df.sort_values('timestamp', ascending=False)

        weights = {}

        for alg in all_algs:
            alg_df = checked_df[checked_df['algorithm'] == alg].head(recent_n)

            if alg_df.empty:
                weights[alg] = 1.0
                continue

            # 지수 감쇠 가중 평균 계산
            hits = alg_df['hit_count'].values
            n = len(hits)

            # 감쇠 가중치: [1, decay, decay^2, ...]
            decay_weights = np.array([decay_factor ** i for i in range(n)])
            decay_weights = decay_weights / decay_weights.sum()

            weighted_avg = np.sum(hits * decay_weights)

            # 트렌드 보너스: 최근 성능이 과거보다 좋으면 가산점
            if n >= 3:
                recent_avg = np.mean(hits[:n//3])
                old_avg = np.mean(hits[2*n//3:])
                trend_bonus = (recent_avg - old_avg) * 0.1
            else:
                trend_bonus = 0

            # 일관성 보너스: 표준편차가 낮으면 신뢰도 높음
            if n >= 2:
                consistency = 1.0 / (1.0 + np.std(hits))
            else:
                consistency = 1.0

            # 최종 가중치 계산
            final_weight = weighted_avg * consistency + trend_bonus

            weights[alg] = max(float(final_weight), 0.1)

        # 정규화 (선택적)
        if weights:
            max_weight = max(weights.values())
            if max_weight > 0:
                weights = {k: v / max_weight * 2.0 for k, v in weights.items()}

        return weights

    def get_algorithm_stats(self) -> Dict[str, Dict]:
        """
        각 알고리즘의 상세 통계 반환.
        UI에서 성능 분석에 사용.
        """
        excluded_algs = [AlgorithmType.ENSEMBLE, AlgorithmType.ULTIMATE]
        all_algs = [alg.value for alg in AlgorithmType if alg not in excluded_algs]

        if not os.path.exists(HISTORY_FILE):
            return {alg: {'count': 0, 'avg': 0, 'max': 0, 'min': 0} for alg in all_algs}

        df = pd.read_csv(HISTORY_FILE)
        checked_df = df[df['hit_count'] != -1]

        stats = {}
        for alg in all_algs:
            alg_df = checked_df[checked_df['algorithm'] == alg]

            if alg_df.empty:
                stats[alg] = {'count': 0, 'avg': 0, 'max': 0, 'min': 0, 'std': 0}
            else:
                hits = alg_df['hit_count'].values
                stats[alg] = {
                    'count': len(hits),
                    'avg': float(np.mean(hits)),
                    'max': int(np.max(hits)),
                    'min': int(np.min(hits)),
                    'std': float(np.std(hits))
                }

        return stats

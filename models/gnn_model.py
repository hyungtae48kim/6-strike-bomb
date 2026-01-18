import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from .base_model import LottoModel
import pandas as pd
import numpy as np
from collections import defaultdict

class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.out = torch.nn.Linear(16, 1) # Output a single score per node

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # GCN layers
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Output layer
        x = self.out(x)
        return torch.sigmoid(x)

class GNNModel(LottoModel):
    """
    Graph Neural Network 기반 로또 예측 모델.
    번호 간 동시 출현 패턴을 그래프로 분석합니다.
    """

    def __init__(self):
        self.model = None
        self.data_graph = None
        self.node_features = None
        self._probability_dist = None
    
    def _build_graph(self, df):
        # Nodes: 0 to 44 (representing 1 to 45)
        num_nodes = 45
        
        # Edges: Co-occurrence
        adj = defaultdict(int)
        node_stats = defaultdict(lambda: {'freq': 0, 'last_seen': 0})
        
        recent_draws = df.sort_values(by='drwNo', ascending=True)
        total_draws = len(recent_draws)
        
        for idx, row in recent_draws.iterrows():
            draw_nums = [row[f'drwtNo{i}'] - 1 for i in range(1, 7)] # 0-indexed
            
            # Update Node Stats
            for n in draw_nums:
                node_stats[n]['freq'] += 1
                node_stats[n]['last_seen'] = row['drwNo']
                
            # Update Edges
            for i in range(len(draw_nums)):
                for j in range(i + 1, len(draw_nums)):
                    u, v = sorted([draw_nums[i], draw_nums[j]])
                    adj[(u, v)] += 1

        # Create Edge Index and Weights for PyG
        edge_index = []
        edge_weights = []
        
        for (u, v), weight in adj.items():
            # Add undirected edges
            edge_index.append([u, v])
            edge_weights.append(weight)
            edge_index.append([v, u])
            edge_weights.append(weight)
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Normalize edge weights
        if edge_weights.max() > 0:
            edge_weights = edge_weights / edge_weights.max()
            
        # Create Node Features
        # Feature 1: Normalized Frequency
        # Feature 2: Recency (Draws since last seen / Total Draws)
        x = []
        current_draw_no = total_draws # Approximate
        for i in range(num_nodes):
            freq = node_stats[i]['freq']
            recency = (current_draw_no - node_stats[i]['last_seen']) if node_stats[i]['last_seen'] > 0 else current_draw_no
            x.append([freq / total_draws, recency / total_draws])
            
        x = torch.tensor(x, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weights)

    def train(self, df: pd.DataFrame):
        print("Building graph for GNN...")
        self.data_graph = self._build_graph(df)
        
        # Model initialization
        self.model = GCN(num_node_features=2) # 2 features: freq, recency
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        # Training loop
        # For simplicity in this demo, we train to predict "hotness" or simple presence based on the graph structure itself
        # Ideally we would do temporal splitting, but let's train to reconstruct probability of being picked
        # based on the constructed graph features. 
        # A true "next draw" predictor would need time-series sliding windows.
        # Here we approximate: Nodes with high centrality/freq in the graph should have high scores? 
        # Actually let's use the recent frequency as a soft label proxy for training to give it something to learn,
        # or just random self-supervised objective?
        
        # Let's try a smarter Target:
        # Target = 1 if the number appeared in the LAST 10 draws (Recent Hot), 0 otherwise.
        # This teaches the GNN to identify "currently active" clusters.
        
        last_10_draws = df.sort_values(by='drwNo', ascending=False).head(10)
        recent_nums = set()
        for i in range(1, 7):
            recent_nums.update(last_10_draws[f'drwtNo{i}'].unique())
            
        y = torch.zeros(45, 1)
        for val in recent_nums:
            y[val-1] = 1.0 # 0-indexed
            
        self.model.train()
        print("Training GNN...")
        for epoch in range(200):
            optimizer.zero_grad()
            out = self.model(self.data_graph)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        print("GNN Training complete.")

        # 확률 분포 계산
        self._compute_probability_distribution()

    def _compute_probability_distribution(self):
        """GNN 스코어 기반 확률 분포 계산"""
        if self.model is None:
            self._probability_dist = np.ones(45) / 45
            return

        self.model.eval()
        with torch.no_grad():
            pred_scores = self.model(self.data_graph).flatten().numpy()

        # Softmax로 확률 분포 변환
        exp_scores = np.exp(pred_scores - np.max(pred_scores))
        self._probability_dist = exp_scores / exp_scores.sum()

    def get_probability_distribution(self) -> np.ndarray:
        """45차원 확률 벡터 반환"""
        if self._probability_dist is None:
            return np.ones(45) / 45
        return self._probability_dist.copy()

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            pred_scores = self.model(self.data_graph).flatten()

        # 상위 6개 인덱스 추출
        _, top_indices = torch.topk(pred_scores, 6)

        # 1-based 인덱싱으로 변환
        prediction = [idx.item() + 1 for idx in top_indices]
        return sorted(prediction)

import sys
import os
import pandas as pd
from models.bayes_model import BayesModel

# Mock data creation if no data exists, or use existing logic
def create_mock_data():
    data = {
        'drwNo': list(range(1, 11)),
        'drwNoDate': pd.date_range(start='2023-01-01', periods=10),
        'drwtNo1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'drwtNo2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'drwtNo3': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'drwtNo4': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        'drwtNo5': [41, 42, 43, 44, 45, 1, 2, 3, 4, 5],
        'drwtNo6': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'bnusNo': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    return pd.DataFrame(data)

def test_bayes_model():
    print("Testing BayesModel...")
    
    # Setup
    df = create_mock_data()
    model = BayesModel()
    
    # Train
    print("Training model...")
    model.train(df)
    
    # Predict
    print("Predicting...")
    prediction = model.predict()
    
    # Verification
    print(f"Prediction: {prediction}")
    
    if len(prediction) != 6:
        print("FAIL: Prediction length is not 6")
        return False
        
    if len(set(prediction)) != 6:
        print("FAIL: Prediction contains duplicates")
        return False
    
    if any(n < 1 or n > 45 for n in prediction):
        print("FAIL: Prediction contains numbers out of range (1-45)")
        return False
        
    print("PASS: BayesModel functionality verified.")
    return True

if __name__ == "__main__":
    success = test_bayes_model()
    if not success:
        sys.exit(1)

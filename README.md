# 6-Strike-Bomb: Korean Lotto 6/45 Predictor

This is a Python-based software to predict Korean Lotto 6/45 winning numbers using basic statistics and Graph Neural Networks (GNN).

## Features
- **Data Fetching**: Automatically fetches the latest draw results from the official source.
- **Algorithms**:
  - **Stats Based**: Uses frequency analysis and "hot" numbers.
  - **GNN Based**: Uses a Graph Convolutional Network to analyze co-occurrence patterns of numbers.
- **UI**: A modern web-based interface (Streamlit) in Korean.

## Installation

1. **Prerequisites**: Python 3.8+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: It is recommended to use a virtual environment.*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

To start the application, run:

```bash
streamlit run app.py
```

This will open a browser window with the User Interface.

## Project Structure
- `data/`: Stores fetched lottery history.
- `models/`: Contains the prediction algorithms.
- `utils/`: Utility scripts (data fetcher).
- `app.py`: Main application entry point.

## Disclaimer
This software is for educational and entertainment purposes only. Lottery numbers are random, and this software does not guarantee winning results.
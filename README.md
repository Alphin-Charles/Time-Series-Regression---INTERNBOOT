# Time Series Regression (Trend + Seasonality)

**Description**

This project demonstrates how to adapt regression models to time-series sales data by capturing both **trend** and **seasonality**.

**Folder structure**

```
time_series_regression/
├── data/
│   └── train.csv                  # Place the dataset here (download manually)
├── notebooks/
│   └── EDA_and_Modeling.ipynb     # Jupyter notebook for exploration
├── main.py                        # Main script to run regression models
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
```

**How to use**

1. Download `train.csv` from the Kaggle competition: https://www.kaggle.com/competitions/store-sales-time-seriesforecasting
2. Place `train.csv` inside the `data/` folder.
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on macOS/Linux
   venv\Scripts\activate    # on Windows (PowerShell)
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```bash
   python main.py
   ```

**Notes**

- `main.py` includes a baseline linear regression and a polynomial trend + seasonality model. It uses simple cyclical encoding for month/weekday and a placeholder holiday indicator (you can replace with actual holiday dates for your country).
- The notebook contains EDA steps and visualizations to help understand trends/seasonality.

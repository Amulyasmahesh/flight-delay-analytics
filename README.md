# ✈️ Flight Delay Analytics & Risk Predictor

End-to-end analysis of **5.7 million US domestic flights (2015)**: exploratory analysis, SQL analytics, a delay-prediction model, and an interactive Streamlit app that returns a delay-risk score for any airline, route, and departure time.

### 🔗 [View the live Tableau dashboard](https://public.tableau.com/app/profile/amulya.m4207/viz/USFlightDelayAnalytics2015/Dashboard1)

![Flight delay dashboard](images/dashboard.png)

## Business Question

> When, where, and with which airline are flights most likely to be delayed, and how well can delay risk be predicted using only information known **before** departure?

A flight is labelled **delayed** if it arrives more than **15 minutes** late (the US DOT standard for on-time performance).

## Key Findings

| Angle | Finding |
|---|---|
| **Overall** | ~17.9% of completed flights arrived 15+ minutes late |
| **Time of day** | Delay rate climbs from ~6.9% at 5 AM to ~25.5% at 8 PM, showing delays cascade through the day |
| **Season** | February (22.5%) and June (22.7%) are the worst months; September (12.4%) and October (11.9%) are the best, nearly a 2x swing |
| **Day of week** | Thursday and Monday are worst (~19%); Saturday is best (~15.4%) |
| **Airline** | Spirit (28.8%) has nearly 3x the delay rate of Hawaiian (10.5%) |
| **Airport** | Chicago O'Hare (23.4% across 276K departures), LaGuardia (23.2%) and JFK (21.4%) rank among the most delay-prone major hubs |

## Approach

1. **Data cleaning:** removed cancelled and diverted flights (~105K rows, all of the missing `ARRIVAL_DELAY` values), leaving 5,714,008 flights.
2. **EDA (pandas, matplotlib):** delay rates by month, weekday, airline, airport, and hour of day.
3. **SQL analytics (SQLite):** ranking and trend analysis using CTEs and window functions (`RANK()`, `LAG()`), including monthly deviation from the yearly average.
4. **Modeling:** Random Forest baseline vs. XGBoost, using only pre-departure features to avoid data leakage.
5. **Dashboard (Tableau Public):** exported small summary tables from the SQL/pandas layer and built an interactive dashboard: KPI cards, delay rate by departure hour, month × weekday heatmap, airline ranking, and a map of origin airports sized by flight volume and coloured by delay rate.
6. **App:** Streamlit interface that loads the trained model and predicts delay probability from user inputs.

## Model Results

Features used: month, day of week, airline, origin, destination, scheduled departure, distance, departure hour, and a rush-hour flag. Leakage columns such as `DEPARTURE_DELAY` were deliberately excluded.

| Model | ROC-AUC | Recall (delayed) | Precision (delayed) |
|---|---|---|---|
| Random Forest | 0.684 | 0.66 | 0.27 |
| XGBoost | 0.687 | 0.66 | 0.27 |

Top predictors (XGBoost): scheduled departure time (34%), airline (16.5%), month (16.4%), day of week (9.4%).

**Takeaway:** both algorithms plateau at roughly the same score. The limit is the feature set, not the algorithm: real-time weather and air-traffic conditions, the strongest real-world causes of delay, are not in this dataset. A weekend flag was also tested and dropped, since day of week already captures that information.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · SQLite · matplotlib · Tableau Public · Streamlit · joblib · Git

## Project Structure

```
flight-delay-analytics/
├── app/
│   └── app.py                  # Streamlit app
├── notebooks/
│   └── 01_exploration.ipynb    # Cleaning, EDA, SQL, modeling
├── src/
│   ├── xgb_model.pkl           # Trained XGBoost model
│   └── label_encoders.pkl      # Fitted encoders for airline/airport codes
├── images/
│   └── dashboard.png           # Dashboard screenshot used in this README
├── data/                       # Not tracked in git (see below)
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

## Getting Started

**1. Clone and set up the environment**
```bash
git clone https://github.com/Amulyasmahesh/flight-delay-analytics.git
cd flight-delay-analytics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost also needs OpenMP: `brew install libomp`

**2. Download the data**

The raw data is too large for GitHub (`flights.csv` is ~565 MB). Download the [2015 Flight Delays and Cancellations dataset](https://www.kaggle.com/datasets/usdot/flight-delays) from Kaggle and place `flights.csv`, `airlines.csv`, and `airports.csv` in `data/raw/`.

**3. Run the notebook** (optional, to reproduce the analysis and retrain the model)

Open `notebooks/01_exploration.ipynb` and run all cells.

**4. Launch the app** (from the project root)
```bash
streamlit run app/app.py
```

## Limitations & Future Work

- **No weather or air-traffic data.** Joining NOAA weather observations by airport and hour is the most promising way to improve the model.
- **Single year (2015).** Newer BTS data would test whether the patterns hold over time.
- **Label encoding** for airports is compact but imposes an arbitrary order; target encoding or embeddings could be tested.
- **Threshold tuning.** The model favors recall (catching delays) over precision; the decision threshold could be tuned to a specific use case.
- **Deployment.** Host the Streamlit app publicly (e.g., Streamlit Community Cloud).

## Author

**Amulya S M**
[LinkedIn](https://linkedin.com/in/amulya-s-m-7b187b253) · [GitHub](https://github.com/Amulyasmahesh)

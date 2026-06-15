import streamlit as st, pandas as pd, numpy as np, json, os, requests, re
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""
st.title("Mini Project B Starter")
name=st.text_input("Student Name")
sid=st.text_input("Student ID")
path=st.text_input("Dataset Path","data/dataset_sample.csv")
df=pd.read_csv(path)
# =========================
# MODELING SECTION
# =========================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Time-based split (important for forecasting)
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Multiple Models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []

for model_name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    r2 = r2_score(y_test, pred)

    results.append({
        "Model": model_name,
        "MAE": round(mae,3),
        "RMSE": round(rmse,3),
        "R2": round(r2,3)
    })

results_df = pd.DataFrame(results)

st.subheader("Model Performance Comparison")
st.dataframe(results_df)

best_model = results_df.sort_values(
    by="RMSE"
).iloc[0]

st.success(
    f"Best Model: {best_model['Model']} | RMSE = {best_model['RMSE']}"
)

# =========================
# DASHBOARD SECTION
# =========================

st.subheader("Power Consumption Trend")

st.line_chart(
    d.set_index(ts)[target]
)

st.subheader("Correlation Matrix")

corr = feat[
    [
        "lag_1",
        "lag_24",
        "rolling_mean_24",
        "hour",
        "month",
        target
    ]
].corr()

st.dataframe(corr)

st.subheader("Dataset Statistics")

st.write(
    feat.describe()
)

st.subheader("Project Insights")

st.markdown("""
### Key Findings

1. Lag features are strongly related to future power demand.

2. Rolling Mean improves forecast stability.

3. Time-based split prevents data leakage.

4. Tree-based models generally outperform Linear Regression.

5. Hour and Month features capture seasonality patterns.

6. Forecasting accuracy can be improved using more advanced models such as XGBoost, LSTM, and GRU.

7. The dataset shows clear temporal patterns suitable for time-series forecasting.
""")

# Display top records

st.subheader("Prepared Dataset")

st.dataframe(
    feat.head(20)
)

# Display metrics chart

st.subheader("RMSE Comparison")

chart_df = results_df.set_index("Model")

# =========================

# MODELING

# =========================

split_idx = int(len(X)*0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

models = {

```
"Linear Regression":
    LinearRegression(),

"Decision Tree":
    DecisionTreeRegressor(
        random_state=42
    ),

"Random Forest":
    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

"Extra Trees":
    ExtraTreesRegressor(
        n_estimators=200,
        random_state=42
    ),

"Gradient Boosting":
    GradientBoostingRegressor(
        random_state=42
    ),

"AdaBoost":
    AdaBoostRegressor(
        random_state=42
    )
```

}

results = []

for model_name, model in models.items():

```
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

r2 = r2_score(
    y_test,
    pred
)

results.append({
    "Model": model_name,
    "MAE": round(mae,3),
    "RMSE": round(rmse,3),
    "R2": round(r2,3)
})
```

results_df = pd.DataFrame(results)

st.subheader("Model Comparison")

st.dataframe(results_df)

best_model = results_df.sort_values(
"RMSE"
).iloc[0]

st.success(
f"Best Model: {best_model['Model']}"
)

    chart_df["RMSE"]

audit=pd.DataFrame({"dtype":df.dtypes.astype(str),"missing_pct":(df.isna().mean()*100)})
st.write(audit)
ts=st.selectbox("Timestamp", df.columns, index=list(df.columns).index("DateTime") if "DateTime" in df.columns else 0)
target=st.selectbox("Target", df.select_dtypes(include=[np.number]).columns, index=list(df.select_dtypes(include=[np.number]).columns).index("Power_kW") if "Power_kW" in df.columns else 0)
h=st.number_input("Horizon",1,168,24)
d=df.copy(); d[ts]=pd.to_datetime(d[ts]); d[target]=pd.to_numeric(d[target],errors="coerce"); d=d.dropna(subset=[ts,target]).sort_values(ts)
res=st.selectbox("Resample",["None","H","D"])
if res!="None": d=d.set_index(ts).resample(res)[target].mean().reset_index()
d["lag_1"]=d[target].shift(1); d["lag_24"]=d[target].shift(24); d["rolling_mean_24"]=d[target].shift(1).rolling(24).mean()
d["hour"]=pd.to_datetime(d[ts]).dt.hour; d["weekend"]=pd.to_datetime(d[ts]).dt.dayofweek>=5; d["month"]=pd.to_datetime(d[ts]).dt.month
d["y_target"]=d[target].shift(-int(h))
feat=d.dropna()
X=feat[["lag_1","lag_24","rolling_mean_24","hour","weekend","month"]]; y=feat["y_target"]
st.write(feat.head())
results_df=None
st.code("# STUDENT ADDITIONS - MODELING")
st.code("# STUDENT ADDITIONS - DASHBOARD")
submission={"student_name":name,"student_id":sid,"has_metrics_table":isinstance(results_df,pd.DataFrame),"results_table":[] if results_df is None else results_df.to_dict(orient="records")}
st.download_button("submission.json",json.dumps(submission,indent=2),"submission.json")
st.download_button("project_card.md",f"# Project\nStudent: {name}","project_card.md")

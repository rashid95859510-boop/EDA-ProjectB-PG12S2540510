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
st.write(df.head(10))
audit=pd.DataFrame({"dtype":df.dtypes.astype(str),"missing_pct":(df.isna().mean()*100)})
st.write(audit)
ts=st.selectbox("Timestamp", df.columns, index=list(df.columns).index("DateTime") if "DateTime" in df.columns else 0)
target=st.selectbox("Target", df.select_dtypes(include=[np.number]).columns, index=list(df.select_dtypes(include=[np.number]).columns).index("Power_kW") if "Power_kW" in df.columns else 0)
h=st.number_input("Horizon",1,168,24)
d=df.copy(); d[ts]=pd.to_datetime(d[ts]); d[target]=pd.to_numeric(d[target],errors="coerce"); d=d.dropna(subset=[ts,target]).sort_values(ts)
res=st.selectbox("Resample",["None","H","D"])
if res!="None": d=d.set_index(ts).resample(res)[target].mean().reset_index()
d["lag_1"] = d[target].shift(1)
d["lag_24"] = d[target].shift(24)

# NEW FEATURES
d["lag_48"] = d[target].shift(48)
d["lag_72"] = d[target].shift(72)
d["lag_168"] = d[target].shift(168)

d["rolling_mean_24"] = (
    d[target]
    .shift(1)
    .rolling(24)
    .mean()
)

d["rolling_std_24"] = (
    d[target]
    .shift(1)
    .rolling(24)
    .std()
)

d["rolling_max_24"] = (
    d[target]
    .shift(1)
    .rolling(24)
    .max()
)

d["rolling_min_24"] = (
    d[target]
    .shift(1)
    .rolling(24)
    .min()
)

d["hour"] = pd.to_datetime(d[ts]).dt.hour
d["dayofweek"] = pd.to_datetime(d[ts]).dt.dayofweek
d["weekend"] = pd.to_datetime(d[ts]).dt.dayofweek >= 5
d["month"] = pd.to_datetime(d[ts]).dt.month
d["quarter"] = pd.to_datetime(d[ts]).dt.quarter
d["y_target"]=d[target].shift(-int(h))
feat=d.dropna()
X = feat[
    [
        "lag_1",
        "lag_24",
        "lag_48",
        "lag_72",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_max_24",
        "rolling_min_24",
        "hour",
        "dayofweek",
        "weekend",
        "month",
        "quarter"
    ]
]
y=feat["y_target"]
st.write(feat.head())

# =========================
# STUDENT ADDITIONS - MODELING
# =========================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Time-based split
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

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
    ),

    "AdaBoost": AdaBoostRegressor(
        random_state=42
    )
}

model_predictions = {}

results = []

for model_name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    model_predictions[model_name] = {
        "model": model,
        "predictions": pred
    }

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

st.subheader("Model Comparison")

st.dataframe(results_df)


st.code("# STUDENT ADDITIONS - DASHBOARD")
submission={"student_name":name,"student_id":sid,"has_metrics_table":isinstance(results_df,pd.DataFrame),"results_table":[] if results_df is None else results_df.to_dict(orient="records")}
st.download_button("submission.json",json.dumps(submission,indent=2),"submission.json")
st.download_button("project_card.md",f"# Project\nStudent: {name}","project_card.md")

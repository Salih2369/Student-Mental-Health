import streamlit as st
import pandas as pd
import joblib

# =============================
# 1) Load Model + Encoder + Data
# =============================
model = joblib.load("anxiety_catboost_model.joblib")
label_encoder = joblib.load("anxiety_label_encoder.joblib")
df = pd.read_csv("clean_data.csv")

# =============================
# 2) Feature Columns (نفس ترتيب التدريب)
# =============================
feature_cols = [
    "gender",
    "age",
    "university",
    "degree_level",
    "degree_major",
    "academic_year",
    "cgpa",
    "residential_status",
    "campus_discrimination",
    "sports_engagement",
    "average_sleep",
    "study_satisfaction",
    "academic_workload",
    "academic_pressure",
    "financial_concerns",
    "social_relationships",
    "depression",
    "isolation",
    "future_insecurity",
    "stress_relief_activities",
]

# =============================
# 3) Streamlit Setup
# =============================
st.set_page_config(page_title="Student Anxiety Prediction App", page_icon="🧠")
st.title("🧠 Student Anxiety Prediction App")
st.write("This app predicts **student anxiety level (Low / Moderate / High)** using a trained ML model.")

st.header("Enter Student Information")

input_data = {}

# =============================
# 4) Numeric Inputs — مع Range ثابت
# =============================

# Age
input_data["age"] = st.slider("Age", 17, 30, 21)

# CGPA → CatBoost يعتبرها Categorical → نخلي input float لكن نرسل له string
input_data["cgpa"] = st.slider("CGPA", 0.0, 4.0, 3.0, step=0.1)

# Sleep hours
input_data["average_sleep"] = st.slider("Sleep Hours", 0.0, 12.0, 6.0, step=0.5)

# Scale 1–5 inputs
input_data["study_satisfaction"] = st.slider("Study Satisfaction", 1, 5, 3)
input_data["academic_workload"] = st.slider("Academic Workload", 1, 5, 3)
input_data["academic_pressure"] = st.slider("Academic Pressure", 1, 5, 3)
input_data["financial_concerns"] = st.slider("Financial Concerns", 1, 5, 3)
input_data["social_relationships"] = st.slider("Social Relationships", 1, 5, 3)
input_data["depression"] = st.slider("Depression", 1, 5, 3)
input_data["isolation"] = st.slider("Isolation", 1, 5, 3)
input_data["future_insecurity"] = st.slider("Future Insecurity", 1, 5, 3)

# =============================
# 5) Categorical Inputs
# =============================
input_data["gender"] = st.selectbox("Gender", df["gender"].dropna().unique())
input_data["university"] = st.selectbox("University", df["university"].dropna().unique())
input_data["degree_level"] = st.selectbox("Degree Level", df["degree_level"].dropna().unique())
input_data["degree_major"] = st.selectbox("Degree Major", df["degree_major"].dropna().unique())
input_data["academic_year"] = st.selectbox("Academic Year", df["academic_year"].dropna().unique())
input_data["residential_status"] = st.selectbox("Residential Status", df["residential_status"].dropna().unique())
input_data["campus_discrimination"] = st.selectbox("Campus Discrimination", df["campus_discrimination"].dropna().unique())
input_data["sports_engagement"] = st.selectbox("Sports Engagement", df["sports_engagement"].dropna().unique())
input_data["stress_relief_activities"] = st.selectbox("Stress Relief Activities", df["stress_relief_activities"].dropna().unique())

# =============================
# 6) Prediction Button
# =============================
if st.button("Predict Anxiety Level"):

    # تحويل إلى DataFrame بنفس ترتيب التدريب
    input_df = pd.DataFrame([input_data], columns=feature_cols)

    # ملء أي NaN
    input_df = input_df.fillna(0)

    # 🔥 CatBoost considers these features categorical during training
    # لذلك يجب أن تكون string وليس float
    input_df["cgpa"] = input_df["cgpa"].astype(str)
    input_df["average_sleep"] = input_df["average_sleep"].astype(str)
    input_df["age"] = input_df["age"].astype(str)

    # بقية الأعمدة الفئوية نحولها strings أيضاً
    categorical_columns = [
        "gender", "university", "degree_level", "degree_major",
        "academic_year", "residential_status", "campus_discrimination",
        "sports_engagement", "stress_relief_activities"
    ]

    for col in categorical_columns:
        input_df[col] = input_df[col].astype(str)

    # 🔥 Predict
    y_pred = model.predict(input_df)

    # Flatten لو لزم
    try:
        y_pred = y_pred.flatten()
    except:
        pass

    class_index = int(y_pred[0])
    result = label_encoder.inverse_transform([class_index])[0]

    st.success(f"Predicted Anxiety Level: **{result}**")

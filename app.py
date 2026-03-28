import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(page_title="Health Monitoring System", layout="wide")

# -----------------------------
# SIDEBAR (PATIENT DETAILS)
# -----------------------------
st.sidebar.title("Patient Details")
st.sidebar.image("1 copy.jpg", width=150)
st.sidebar.write("Name: Priyaranjan Show")
st.sidebar.write("Age: 21")
st.sidebar.write("Condition: Live Monitoring")

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 Intelligent Health Monitoring System")
st.subheader("AI-based Silent Heart Attack & Fall Detection")

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("health_data.csv")

# ✅ CLEAN + FIX COLUMN NAMES
data.columns = data.columns.str.strip().str.lower()

# Rename to match your code
data.rename(columns={
    'heartrate': 'heart_rate',
    'temperature': 'temp',
    'acceleration': 'fall'
}, inplace=True)

# Debug check (optional)
st.write("Dataset Columns:", data.columns)

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = data[['heart_rate', 'spo2', 'temp', 'fall', 'ecg']]
y = data['risk']

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X, y)

# -----------------------------
# GRAPH SECTION
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ❤️ Heart Rate")
    hr_chart = st.line_chart()

with col2:
    st.markdown("### 🫁 SpO2")
    spo2_chart = st.line_chart()

with col3:
    st.markdown("### 🌡 Temperature")
    temp_chart = st.line_chart()

# -----------------------------
# FUNCTIONS
# -----------------------------
def get_status(risk, hr, spo2, temp, fall):
    if spo2 < 90 or hr > 120 or temp > 38.5:
        return "CRITICAL"
    elif risk == 1:
        return "HIGH RISK"
    elif fall == 1:
        return "FALL ALERT"
    else:
        return "NORMAL"

def advice(status):
    if status == "CRITICAL":
        return "🚨 Immediate medical attention required!"
    elif status == "HIGH RISK":
        return "⚠ Consult doctor urgently"
    elif status == "FALL ALERT":
        return "⚠ Check patient immediately"
    else:
        return "✅ Patient stable"

# -----------------------------
# REAL-TIME MONITORING
# -----------------------------
st.subheader("Live Monitoring")

placeholder = st.empty()
report_data = []

for i in range(len(data)):

    row = data.iloc[i]

    hr = row['heart_rate']
    spo2 = row['spo2']
    temp = row['temp']
    fall = row['fall']
    ecg = row['ecg']

    # ✅ Prediction (fixed)
    input_data = np.array([[hr, spo2, temp, fall, ecg]])

    risk = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    risk_percent = int(prob * 100)

    status = get_status(risk, hr, spo2, temp, fall)
    msg = advice(status)

    report_data.append([hr, spo2, temp, fall, ecg, risk_percent, status])

    # -----------------------------
    # DISPLAY
    # -----------------------------
    with placeholder.container():

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("❤️ HR", round(hr,2))
        c2.metric("🫁 SpO2", round(spo2,2))
        c3.metric("🌡 Temp", round(temp,2))
        c4.metric("🚶 Fall", "Yes" if fall==1 else "No")

        st.write(f"### Risk Score: {risk_percent}%")
        st.progress(risk_percent/100)

        if status == "CRITICAL":
            st.error("🚨 CRITICAL CONDITION")
        elif status == "HIGH RISK":
            st.warning("⚠ HIGH RISK")
        elif status == "FALL ALERT":
            st.warning("⚠ FALL DETECTED")
        else:
            st.success("✅ NORMAL")

        st.info(msg)

    # -----------------------------
    # GRAPH UPDATE
    # -----------------------------
    hr_chart.add_rows(pd.DataFrame({"Heart Rate":[hr]}))
    spo2_chart.add_rows(pd.DataFrame({"SpO2":[spo2]}))
    temp_chart.add_rows(pd.DataFrame({"Temperature":[temp]}))

    time.sleep(1.2)

# -----------------------------
# SAVE REPORT
# -----------------------------
report_df = pd.DataFrame(
    report_data,
    columns=["HR","SpO2","Temp","Fall","ECG","Risk%","Status"]
)

report_df.to_csv("patient_report.csv", index=False)

st.success("✅ Report saved as patient_report.csv")

# DOWNLOAD BUTTON
with open("patient_report.csv", "rb") as file:
    st.download_button("📥 Download Report", file, "patient_report.csv")

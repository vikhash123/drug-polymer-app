import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("DoE_Experimental_Data.csv")

# Encode categorical variables
label_encoders = {}
for col in ["Drug", "Polymer", "Ratio", "Method"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df["Compatible"] = df["Compatible"].map({"Yes": 1, "No": 0})

# Train model
X = df.drop("Compatible", axis=1)
y = df["Compatible"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit app
st.title("🧪 Drug–Polymer Compatibility Predictor")
st.write("Enter formulation parameters to get AI-based compatibility prediction.")

# Input fields
drug = st.selectbox("Select Drug", label_encoders["Drug"].classes_)
logp = st.number_input("LogP")
mw = st.number_input("Molecular Weight")
polymer = st.selectbox("Select Polymer", label_encoders["Polymer"].classes_)
ratio = st.selectbox("Select Ratio", label_encoders["Ratio"].classes_)
method = st.selectbox("Select Method", label_encoders["Method"].classes_)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, step=0.1)

if st.button("Predict Compatibility"):
    input_data = {
        "Drug": label_encoders["Drug"].transform([drug])[0],
        "LogP": logp,
        "MW": mw,
        "Polymer": label_encoders["Polymer"].transform([polymer])[0],
        "Ratio": label_encoders["Ratio"].transform([ratio])[0],
        "Method": label_encoders["Method"].transform([method])[0],
        "pH": ph
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Compatible")
    else:
        st.error("❌ Not Compatible")

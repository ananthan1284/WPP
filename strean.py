
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Water Potability Predictor", layout="wide")

st.title("💧 Water Potability Prediction App")

# Canvas Section
st.subheader("Draw or Annotate (Canvas Demo)")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=2,
    stroke_color="#000000",
    background_color="#eee",
    height=200,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, caption="Your Drawing")
# Upload or load data
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload water_potability.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("water_potability.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Data Cleaning
st.subheader("Missing Values Heatmap")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
st.pyplot(fig)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Features and Target
X = df_imputed.drop("Potability", axis=1)
y = df_imputed["Potability"]

# Oversampling
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.3, random_state=42)

# Sidebar: Model Selection
model_name = st.sidebar.selectbox("Choose Classifier", ("Random Forest", "SVM", "XGBoost"))

if model_name == "Random Forest":
    model = RandomForestClassifier()
elif model_name == "SVM":
    model = SVC(probability=True)
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train and evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# User Prediction Input
st.sidebar.header("Predict Water Potability")

def user_input_features():
    ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
    Hardness = st.sidebar.slider("Hardness", 50.0, 300.0, 150.0)
    Solids = st.sidebar.slider("Solids", 200.0, 50000.0, 15000.0)
    Chloramines = st.sidebar.slider("Chloramines", 0.0, 15.0, 7.0)
    Sulfate = st.sidebar.slider("Sulfate", 100.0, 500.0, 333.0)
    Conductivity = st.sidebar.slider("Conductivity", 100.0, 800.0, 400.0)
    Organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 30.0, 15.0)
    Trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 120.0, 60.0)
    Turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 5.0)

    data = {
        'ph': ph,
        'Hardness': Hardness,
        'Solids': Solids,
        'Chloramines': Chloramines,
        'Sulfate': Sulfate,
        'Conductivity': Conductivity,
        'Organic_carbon': Organic_carbon,
        'Trihalomethanes': Trihalomethanes,
        'Turbidity': Turbidity
    }
    return pd.DataFrame([data])

user_input = user_input_features()
scaled_input = scaler.transform(user_input)
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

st.subheader("Prediction Result")
st.write("Potable" if prediction[0] == 1 else "Not Potable")
st.subheader("Prediction Probability")
st.write(f"Probability of being potable: {prediction_proba[0][1]:.2f}")
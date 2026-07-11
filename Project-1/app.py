"""
House Price Prediction — Interactive Streamlit App
Converted from House_Price_Prediction.ipynb

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    "datasets/housing/housing.csv"
)

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if file is not None:
        return pd.read_csv(file)
    return pd.read_csv(DEFAULT_DATA_URL)


@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].mean())

    le = LabelEncoder()
    df["ocean_proximity"] = le.fit_transform(df["ocean_proximity"])

    return df, le


@st.cache_resource
def train_model(df: pd.DataFrame, model_name: str, test_size: float, random_state: int = 42):
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if model_name == "LightGBM":
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state)
    else:
        model = DecisionTreeRegressor(random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "train_r2": model.score(X_train, y_train),
        "test_r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    return model, metrics, X.columns.tolist(), (X_test, y_test, y_pred)


# ---------------------------------------------------------------------------
# Sidebar — data + model controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload housing.csv (optional)", type=["csv"])
st.sidebar.caption(
    "No file uploaded? App will auto-download the classic California "
    "housing dataset (same one used in the notebook)."
)

model_name = st.sidebar.selectbox("Model", ["LightGBM", "Decision Tree"], index=0)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)

# ---------------------------------------------------------------------------
# Load + preprocess
# ---------------------------------------------------------------------------
st.title("🏠 House Price Prediction")
st.caption("Interactive version of the House_Price_Prediction notebook")

try:
    raw_df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not load data: {e}\n\nPlease upload housing.csv manually from the sidebar.")
    st.stop()

df, label_encoder = preprocess(raw_df)
ocean_categories = list(label_encoder.classes_)

with st.spinner(f"Training {model_name}..."):
    model, metrics, feature_cols, test_data = train_model(df, model_name, test_size)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_explore, tab_performance = st.tabs(
    ["🔮 Predict", "📊 Explore Data", "📈 Model Performance"]
)

# ---------------- Predict tab ----------------
with tab_predict:
    st.subheader("Enter house details")

    col1, col2, col3 = st.columns(3)
    with col1:
        longitude = st.number_input("Longitude", value=float(raw_df["longitude"].median()), format="%.4f")
        latitude = st.number_input("Latitude", value=float(raw_df["latitude"].median()), format="%.4f")
        housing_median_age = st.number_input(
            "Housing Median Age", value=float(raw_df["housing_median_age"].median())
        )
    with col2:
        total_rooms = st.number_input("Total Rooms", value=int(raw_df["total_rooms"].median()), step=1)
        total_bedrooms = st.number_input(
            "Total Bedrooms", value=int(raw_df["total_bedrooms"].median(skipna=True)), step=1
        )
        population = st.number_input("Population", value=float(raw_df["population"].median()))
    with col3:
        households = st.number_input("Households", value=int(raw_df["households"].median()), step=1)
        median_income = st.number_input(
            "Median Income (10k USD)", value=float(raw_df["median_income"].median()), format="%.4f"
        )
        ocean_proximity = st.selectbox("Ocean Proximity", ocean_categories)

    if st.button("Predict Price", type="primary"):
        ocean_encoded = label_encoder.transform([ocean_proximity])[0]

        new_house = pd.DataFrame({
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity": [ocean_encoded],
        })[feature_cols]

        prediction = model.predict(new_house)[0]
        st.success(f"### Predicted House Price: ${prediction:,.2f}")

# ---------------- Explore tab ----------------
with tab_explore:
    st.subheader("Dataset preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Missing values")
        st.dataframe(raw_df.isnull().sum().rename("missing_count"))
    with c2:
        st.write("Shape")
        st.write(f"{raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

    st.subheader("Median house value distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(raw_df["median_house_value"], bins=50, ax=ax, color="#4C72B0")
    ax.set_xlabel("median_house_value")
    st.pyplot(fig)

    st.subheader("Correlation heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Geographic price map")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    scatter = ax3.scatter(
        raw_df["longitude"], raw_df["latitude"],
        c=raw_df["median_house_value"], cmap="viridis", alpha=0.4, s=10
    )
    plt.colorbar(scatter, ax=ax3, label="median_house_value")
    ax3.set_xlabel("longitude")
    ax3.set_ylabel("latitude")
    st.pyplot(fig3)

# ---------------- Performance tab ----------------
with tab_performance:
    st.subheader(f"{model_name} performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("Train R²", f"{metrics['train_r2']:.4f}")
    m2.metric("Test R²", f"{metrics['test_r2']:.4f}")
    m3.metric("MAE", f"${metrics['mae']:,.0f}")

    X_test, y_test, y_pred = test_data
    st.subheader("Predicted vs Actual")
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    ax4.scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax4.plot(lims, lims, "r--", lw=1)
    ax4.set_xlabel("Actual")
    ax4.set_ylabel("Predicted")
    st.pyplot(fig4)

    if model_name == "LightGBM":
        st.subheader("Feature importance")
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=importance_df, x="importance", y="feature", ax=ax5, color="#55A868")
        st.pyplot(fig5)

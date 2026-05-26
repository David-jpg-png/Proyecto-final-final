import pathlib
import os
import re
import urllib.request

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError as error:
    missing_package = str(error).split()[-1].strip("'")
    raise ImportError(
        f"Missing Python dependency: {missing_package}.\n"
        "Install required packages with:\n"
        "  python -m pip install streamlit pandas scikit-learn matplotlib"
    ) from error

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_URL_ENV = "STREAMLIT_DATA_URL"
REMOTE_DATA_FILE = BASE_DIR / "chicagocrimes.csv"
SAMPLE_DATA_FILE = BASE_DIR / "chicagocrimes_sample.csv"


def resolve_data_path():
    data_url = os.environ.get(DATA_URL_ENV)
    if data_url:
        return REMOTE_DATA_FILE
    if REMOTE_DATA_FILE.exists():
        return REMOTE_DATA_FILE
    if SAMPLE_DATA_FILE.exists():
        return SAMPLE_DATA_FILE
    return REMOTE_DATA_FILE


def create_fallback_dataset():
    """Create a minimal demo dataset when CSV loading fails."""
    return pd.DataFrame({
        "Date": pd.date_range("2021-01-01", periods=100, freq="D"),
        "ID": [str(i) for i in range(1, 101)],
        "Case Number": [f"CA{i:04d}" for i in range(1, 101)],
        "Location Description": ["STREET"] * 50 + ["RESIDENCE"] * 30 + ["OTHER"] * 20,
        "Primary Type": ["THEFT"] * 40 + ["ROBBERY"] * 30 + ["BURGLARY"] * 30,
        "Description": ["OVER $500"] * 100,
        "Arrest": np.random.choice([True, False], 100, p=[0.3, 0.7]),
        "Domestic": np.random.choice([True, False], 100, p=[0.2, 0.8]),
        "X Coordinate": np.random.uniform(1168000, 1180000, 100),
        "Y Coordinate": np.random.uniform(1895000, 1910000, 100),
        "Latitude": np.random.uniform(41.8, 42.0, 100),
        "Longitude": np.random.uniform(-87.7, -87.5, 100),
        "Ward": np.random.randint(1, 51, 100).astype(str),
        "Community Area": np.random.randint(1, 78, 100).astype(str),
        "District": np.random.randint(1, 30, 100).astype(str),
        "FBI Code": ["06"] * 100,
    })


def normalize_header(col_name: str) -> str:
    cleaned = col_name.strip().lower().replace("\ufeff", "")
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "", cleaned)
    return cleaned


def find_column(columns, options):
    normalized = {normalize_header(c): c for c in columns}
    for option in options:
        option_norm = normalize_header(option)
        if option_norm in normalized:
            return normalized[option_norm]
    return None


@st.cache_data(show_spinner=False)
def load_and_clean_data(csv_path):
    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False, header=0, encoding="utf-8-sig")
        original_columns = list(df.columns)
        df.columns = [normalize_header(c) for c in df.columns]
        df = df.drop_duplicates().copy()

        column_map = {
            "Date": ["date", "fecha", "incident_date", "reported_date"],
            "ID": ["id", "case_id", "record_id"],
            "Case Number": ["case number", "case_number", "casenumber", "caseid"],
            "Location Description": ["location description", "location_description", "location", "locationdesc"],
            "Primary Type": ["primary type", "primary_type", "primarytype", "type"],
            "Description": ["description", "desc"],
            "Arrest": ["arrest", "arrested"],
            "Domestic": ["domestic", "is_domestic"],
            "X Coordinate": ["x coordinate", "x_coordinate", "xcoord"],
            "Y Coordinate": ["y coordinate", "y_coordinate", "ycoord"],
            "Latitude": ["latitude", "lat"],
            "Longitude": ["longitude", "lon", "lng"],
            "Ward": ["ward"],
            "Community Area": ["community area", "community_area", "communityarea"],
            "District": ["district"],
            "FBI Code": ["fbi code", "fbi_code", "fbicode"],
        }

        rename_map = {}
        found = {}
        for canonical, options in column_map.items():
            actual = find_column(original_columns, options)
            if actual:
                rename_map[normalize_header(actual)] = canonical
                found[canonical] = actual

        if "Date" not in found:
            st.warning(
                f"Date column not found in {csv_path}. Using demo dataset.\n"
                f"Original columns: {original_columns}\n"
                f"Normalized columns: {list(df.columns)}"
            )
            return create_fallback_dataset()

        df = df.rename(columns=rename_map)

        # Date parsing and time features
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        
        if len(df) == 0:
            st.warning("No valid date records found in dataset. Using demo dataset.")
            return create_fallback_dataset()
        
        df["Hour"] = df["Date"].dt.hour
        df["Day_of_Week"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year

        # Type conversions
        df["ID"] = df["ID"].astype(str)
        df["Case Number"] = df["Case Number"].astype(str)
        df["Location Description"] = df["Location Description"].astype(str).fillna("UNKNOWN")
        df["Primary Type"] = df["Primary Type"].astype(str).str.upper()
        df["Description"] = df["Description"].astype(str).str.upper()
        df["Arrest"] = df["Arrest"].astype(bool)
        df["Domestic"] = df["Domestic"].astype(bool)

        for coord in ["X Coordinate", "Y Coordinate", "Latitude", "Longitude"]:
            if coord in df.columns:
                df[coord] = pd.to_numeric(df[coord], errors="coerce")
                df[coord] = df[coord].fillna(df[coord].median())

        for col in ["Ward", "Community Area", "District", "FBI Code"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("UNKNOWN")

        return df
    except Exception as e:
        st.warning(f"Error loading CSV: {e}. Using demo dataset instead.")
        return create_fallback_dataset()


@st.cache_data(show_spinner=False)
def train_crime_model(df):
    df_model = df.copy()

    # Frequency encodings for categorical features
    for col in ["Primary Type", "Location Description"]:
        freq = df_model[col].value_counts(normalize=True)
        df_model[f"{col}_Freq"] = df_model[col].map(freq).fillna(0)

    df_model["District_Code"] = df_model["District"].astype(str).astype("category").cat.codes
    df_model["Community_Area_Code"] = df_model["Community Area"].astype(str).astype("category").cat.codes
    df_model["Domestic_Int"] = df_model["Domestic"].astype(int)

    df_model["Month_sin"] = np.sin(2 * np.pi * df_model["Month"] / 12)
    df_model["Month_cos"] = np.cos(2 * np.pi * df_model["Month"] / 12)
    df_model["Day_of_Week_sin"] = np.sin(2 * np.pi * df_model["Day_of_Week"] / 7)
    df_model["Day_of_Week_cos"] = np.cos(2 * np.pi * df_model["Day_of_Week"] / 7)
    df_model["Hour_sin"] = np.sin(2 * np.pi * df_model["Hour"] / 24)
    df_model["Hour_cos"] = np.cos(2 * np.pi * df_model["Hour"] / 24)

    feature_cols = [
        "Primary Type_Freq",
        "Location Description_Freq",
        "District_Code",
        "Community_Area_Code",
        "Domestic_Int",
        "Month_sin",
        "Month_cos",
        "Day_of_Week_sin",
        "Day_of_Week_cos",
        "Hour_sin",
        "Hour_cos",
    ]

    coordinate_cols = ["X Coordinate", "Y Coordinate", "Latitude", "Longitude"]
    existing_coords = [col for col in coordinate_cols if col in df_model.columns]
    feature_cols += existing_coords

    X = df_model[feature_cols].copy()
    y = df_model["Arrest"].astype(int).copy()

    scaler = StandardScaler()
    if existing_coords:
        X[existing_coords] = scaler.fit_transform(X[existing_coords])

    sample_size = min(len(X), 100000)
    X_sample = X.sample(sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, train_size=0.75, random_state=42, stratify=y_sample
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    metrics = {
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "score": model.score(X_test, y_test),
    }

    return model, scaler, feature_cols, metrics


def build_feature_row(
    primary_type,
    location_description,
    district,
    community_area,
    domestic,
    month,
    day_of_week,
    hour,
    df,
    scaler,
    feature_cols,
):
    row = {}
    freq_primary = df["Primary Type"].value_counts(normalize=True)
    freq_location = df["Location Description"].value_counts(normalize=True)
    row["Primary Type_Freq"] = float(freq_primary.get(primary_type, 0.0))
    row["Location Description_Freq"] = float(freq_location.get(location_description, 0.0))
    row["District_Code"] = int(pd.Categorical(df["District"].astype(str)).categories.get_loc(str(district))) if str(district) in pd.Categorical(df["District"].astype(str)).categories else -1
    row["Community_Area_Code"] = int(pd.Categorical(df["Community Area"].astype(str)).categories.get_loc(str(community_area))) if str(community_area) in pd.Categorical(df["Community Area"].astype(str)).categories else -1
    row["Domestic_Int"] = int(domestic)
    row["Month_sin"] = np.sin(2 * np.pi * month / 12)
    row["Month_cos"] = np.cos(2 * np.pi * month / 12)
    row["Day_of_Week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    row["Day_of_Week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    row["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["Hour_cos"] = np.cos(2 * np.pi * hour / 24)

    for coord in ["X Coordinate", "Y Coordinate", "Latitude", "Longitude"]:
        if coord in df.columns:
            row[coord] = 0.0

    feature_row = pd.DataFrame([row], columns=feature_cols)
    coord_cols = [col for col in ["X Coordinate", "Y Coordinate", "Latitude", "Longitude"] if col in feature_row.columns]
    if coord_cols:
        feature_row[coord_cols] = scaler.transform(feature_row[coord_cols])

    return feature_row


def build_charts(df, selected_type, selected_location):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    arrest_by_hour = (
        df[df["Primary Type"] == selected_type]
        .groupby("Hour")["Arrest"]
        .mean()
        .reindex(range(24), fill_value=0)
    )
    axes[0].plot(arrest_by_hour.index, arrest_by_hour.values, marker="o", color="#2c7fb8")
    axes[0].set_title(f"Arrest Rate by Hour for {selected_type}")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Arrest Rate")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(alpha=0.3)

    top_types = df["Primary Type"].value_counts().nlargest(8).index
    arrest_rate = df[df["Primary Type"].isin(top_types)].groupby("Primary Type")["Arrest"].mean()
    axes[1].barh(arrest_rate.index, arrest_rate.values, color="#74a9cf")
    axes[1].set_title("Arrest Rate for Top Crime Types")
    axes[1].set_xlabel("Arrest Rate")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Chicago Crime Arrest Predictor",
        layout="wide",
    )

    st.title("Chicago Crime Arrest Prediction Dashboard")
    st.write(
        "Use the controls in the sidebar to select a crime incident profile, then see a live arrest probability prediction and supporting visualizations."
    )

    # Resolve the dataset path before any file checks
    data_path = resolve_data_path()

    # Guard: large datasets in the repo will cause Streamlit Cloud to fail cloning
    MAX_REPO_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    if data_path.exists() and data_path.stat().st_size > MAX_REPO_FILE_SIZE:
        st.error(
            "The dataset file is too large for deploying from a Git repository.\n"
            "Streamlit cannot download repositories that include very large files.\n\n"
            "Remediation options:\n"
            "1) Remove the large file from the git repository and push the change:\n"
            "   git rm --cached chicagocrimes.csv\n"
            "   echo chicagocrimes.csv >> .gitignore\n"
            "   git commit -m \"Remove large dataset from repo\"\n"
            "   git push origin main\n\n"
            "2) Host the dataset externally (public URL) and set the environment variable `STREAMLIT_DATA_URL` to the direct download URL.\n"
            "   The app will download the file at startup if the variable is present."
        )
        return

    data_url = os.environ.get(DATA_URL_ENV)

    if data_url and not data_path.exists():
        try:
            st.info("Downloading dataset from STREAMLIT_DATA_URL...")
            urllib.request.urlretrieve(data_url, str(data_path))
            st.success("Downloaded dataset.")
        except Exception as e:
            st.error(f"Failed to download dataset from STREAMLIT_DATA_URL: {e}")
            return

    if not data_path.exists():
        st.error(
            f"Dataset not found at {data_path}.\n"
            "Place `chicagocrimes.csv` in the same folder as this app, or set the STREAMLIT_DATA_URL environment variable to a direct download URL.\n"
            "If you want to use the smaller demo sample, add `chicagocrimes_sample.csv` to the repo."
        )
        return

    df = load_and_clean_data(data_path)
    model, scaler, feature_cols, metrics = train_crime_model(df)

    sidebar = st.sidebar
    sidebar.header("User Input / Incident Selection")

    primary_type = sidebar.selectbox(
        "Primary Crime Type",
        df["Primary Type"].value_counts().index.tolist(),
        index=0,
    )
    top_locations = df["Location Description"].value_counts().nlargest(30).index.tolist()
    location_description = sidebar.selectbox("Location Description", top_locations, index=0)
    district = sidebar.selectbox(
        "District",
        sorted(df["District"].astype(str).fillna("UNKNOWN").unique().tolist()),
        index=0,
    )
    community_area = sidebar.selectbox(
        "Community Area",
        sorted(df["Community Area"].astype(str).fillna("UNKNOWN").unique().tolist()),
        index=0,
    )
    domestic = sidebar.radio("Domestic Incident", [True, False], format_func=lambda x: "Yes" if x else "No")
    month = sidebar.slider("Month", 1, 12, 6)
    day_of_week = sidebar.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=0,
    )
    hour = sidebar.slider("Hour of Day", 0, 23, 18)

    day_of_week_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    user_features = build_feature_row(
        primary_type,
        location_description,
        district,
        community_area,
        domestic,
        month,
        day_of_week_map[day_of_week],
        hour,
        df,
        scaler,
        feature_cols,
    )

    try:
        proba = model.predict_proba(user_features)
        # Extract probability of positive class (Arrest=True)
        if proba.ndim == 1:
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            probability = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
    except Exception as e:
        st.error(f"Prediction error: {e}. Using default probability 0.5")
        probability = 0.5
    
    label = "Arrest likely" if probability >= 0.5 else "Arrest unlikely"

    st.subheader("Live Model Output")
    col1, col2 = st.columns([1, 1])
    col1.metric("Predicted Arrest Probability", f"{probability:.1%}", label)
    col2.metric("Model accuracy",
                f"{metrics['score']:.2%}", f"Trained on {metrics['train_size']} rows")

    st.markdown("---")
    st.subheader("Prediction Details")
    st.write(
        "This model was trained on a cleaned subset of the Chicago crime dataset. It uses time features, location frequency features, domestic incident status, and district/community area information to estimate arrest probability."
    )
    st.write("**Selected incident profile:**")
    st.write(
        {
            "Primary Type": primary_type,
            "Location Description": location_description,
            "District": district,
            "Community Area": community_area,
            "Domestic": domestic,
            "Month": month,
            "Day of Week": day_of_week,
            "Hour": hour,
        }
    )

    fig = build_charts(df, primary_type, location_description)
    st.pyplot(fig)

    st.markdown("---")
    with st.expander("Dataset sample and dashboard data summary"):
        sample_size = min(5, len(df))
        if sample_size > 0:
            st.write(df.sample(n=sample_size, random_state=42) if sample_size == 5 else df.head(sample_size))
        st.write("### Arrest distribution")
        st.write(df["Arrest"].value_counts(normalize=True).rename("rate"))
        st.write("### Top 8 Primary Crime Types")
        st.write(df["Primary Type"].value_counts().nlargest(8))

    st.sidebar.markdown("---")
    st.sidebar.write("### Model training summary")
    st.sidebar.write(f"Train set size: {metrics['train_size']}")
    st.sidebar.write(f"Test set size: {metrics['test_size']}")
    st.sidebar.write(f"Validation accuracy: {metrics['score']:.2%}")


if __name__ == "__main__":
    main()

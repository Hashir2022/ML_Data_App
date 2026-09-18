import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ML Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = []

if "target_column" not in st.session_state:
    st.session_state.target_column = None

if "task" not in st.session_state:
    st.session_state.task = None

if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None

if "model_name" not in st.session_state:
    st.session_state.model_name = None


# =========================================================
# FUNCTIONS
# =========================================================

def detect_task(y):
    """Automatically detect Classification or Regression."""

    if y.dtype == "object":
        return "Classification"

    if str(y.dtype).startswith("category"):
        return "Classification"

    if y.nunique() <= 10:
        return "Classification"

    return "Regression"


def prepare_features(df, target_column):
    """Prepare dataset features for ML."""

    X = df.drop(
        columns=[target_column]
    ).copy()

    # Remove completely empty columns
    X = X.dropna(
        axis=1,
        how="all"
    )

    # Numeric columns
    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = X[column].fillna(
            median_value
        )

    # Categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:

        mode = X[column].mode()

        if len(mode) > 0:
            fill_value = mode.iloc[0]
        else:
            fill_value = "Unknown"

        X[column] = X[column].fillna(
            fill_value
        )

    # One-hot encoding
    X = pd.get_dummies(
        X,
        drop_first=False
    )

    # Convert bool to numeric
    X = X.astype(float)

    return X


def create_model(task, model_name):

    if task == "Classification":

        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=2000
            )

        if model_name == "Decision Tree":
            return DecisionTreeClassifier(
                random_state=42
            )

        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

    else:

        if model_name == "Linear Regression":
            return LinearRegression()

        if model_name == "Decision Tree":
            return DecisionTreeRegressor(
                random_state=42
            )

        if model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

    return None


def save_model():

    package = {
        "model": st.session_state.model,
        "feature_columns": st.session_state.feature_columns,
        "target_column": st.session_state.target_column,
        "task": st.session_state.task,
        "label_encoder": st.session_state.label_encoder,
        "model_name": st.session_state.model_name
    }

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        package,
        "models/model.pkl"
    )


# =========================================================
# HEADER
# =========================================================

st.title("📊 ML Data Analysis & Prediction Dashboard")

st.markdown(
    """
    Upload your dataset, explore the data,
    visualize patterns, train machine learning models,
    and make predictions.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Dashboard",
        "📂 Upload Dataset",
        "🔍 Data Analysis",
        "🤖 Train Model",
        "🔮 Prediction"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    if st.session_state.df is None:

        st.info(
            "No dataset uploaded yet."
        )

        st.markdown(
            """
            ### Features

            📂 **Upload Dataset**

            Upload CSV datasets.

            🔍 **Data Analysis**

            View dataset information,
            missing values and statistics.

            📊 **Visualization**

            Create interactive charts.

            🤖 **Machine Learning**

            Train classification or regression models.

            🔮 **Prediction**

            Enter feature values and generate predictions.
            """
        )

    else:

        df = st.session_state.df

        st.success(
            "Dataset loaded successfully!"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


# =========================================================
# UPLOAD DATASET
# =========================================================

elif page == "📂 Upload Dataset":

    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.session_state.df = df

            # Reset model
            st.session_state.model = None
            st.session_state.feature_columns = []
            st.session_state.target_column = None
            st.session_state.task = None
            st.session_state.label_encoder = None
            st.session_state.model_name = None

            st.success(
                "Dataset uploaded successfully!"
            )

            st.subheader("Preview")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Rows",
                    df.shape[0]
                )

            with col2:
                st.metric(
                    "Columns",
                    df.shape[1]
                )

            with col3:
                st.metric(
                    "Missing Values",
                    int(df.isnull().sum().sum())
                )

        except Exception as e:

            st.error(
                f"Unable to read CSV: {e}"
            )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "🔍 Data Analysis":

    st.header("🔍 Data Analysis")

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        # -------------------------------------------------
        # OVERVIEW
        # -------------------------------------------------

        st.subheader("📋 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicates",
                int(df.duplicated().sum())
            )

        # -------------------------------------------------
        # DATA PREVIEW
        # -------------------------------------------------

        st.subheader("👀 Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )

        # -------------------------------------------------
        # COLUMN INFORMATION
        # -------------------------------------------------

        st.subheader("📑 Column Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing": [
                int(df[col].isnull().sum())
                for col in df.columns
            ],
            "Unique": [
                int(df[col].nunique())
                for col in df.columns
            ]
        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        st.subheader("📊 Statistical Summary")

        st.dataframe(
            df.describe(
                include="all"
            ).T,
            use_container_width=True
        )

        # -------------------------------------------------
        # MISSING VALUES
        # -------------------------------------------------

        st.subheader("❓ Missing Values")

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": [
                int(df[col].isnull().sum())
                for col in df.columns
            ],
            "Percentage": [
                round(
                    df[col].isnull().mean() * 100,
                    2
                )
                for col in df.columns
            ]
        })

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        # -------------------------------------------------
        # VISUALIZATION
        # -------------------------------------------------

        st.subheader("📈 Data Visualization")

        chart_type = st.selectbox(
            "Choose Visualization",
            [
                "Histogram",
                "Bar Chart",
                "Scatter Plot",
                "Line Chart",
                "Pie Chart",
                "Correlation Heatmap"
            ]
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        all_columns = df.columns.tolist()

        # Histogram
        if chart_type == "Histogram":

            if len(numeric_columns) == 0:

                st.warning(
                    "No numeric columns available."
                )

            else:

                column = st.selectbox(
                    "Select Column",
                    numeric_columns
                )

                fig = px.histogram(
                    df,
                    x=column,
                    title=f"Distribution of {column}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # Bar Chart
        elif chart_type == "Bar Chart":

            column = st.selectbox(
                "Select Column",
                all_columns
            )

            counts = (
                df[column]
                .astype(str)
                .value_counts()
                .head(20)
                .reset_index()
            )

            counts.columns = [
                "Value",
                "Count"
            ]

            fig = px.bar(
                counts,
                x="Value",
                y="Count",
                title=f"Bar Chart - {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Scatter
        elif chart_type == "Scatter Plot":

            if len(numeric_columns) < 2:

                st.warning(
                    "At least two numeric columns are required."
                )

            else:

                x_column = st.selectbox(
                    "X Axis",
                    numeric_columns
                )

                y_column = st.selectbox(
                    "Y Axis",
                    numeric_columns,
                    index=1
                )

                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    title=f"{x_column} vs {y_column}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # Line Chart
        elif chart_type == "Line Chart":

            if not numeric_columns:

                st.warning(
                    "No numeric columns available."
                )

            else:

                column = st.selectbox(
                    "Select Column",
                    numeric_columns
                )

                fig = px.line(
                    df,
                    y=column,
                    title=f"Line Chart - {column}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # Pie Chart
        elif chart_type == "Pie Chart":

            column = st.selectbox(
                "Select Column",
                all_columns
            )

            counts = (
                df[column]
                .astype(str)
                .value_counts()
                .head(10)
            )

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                title=f"Pie Chart - {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Correlation
        elif chart_type == "Correlation Heatmap":

            if len(numeric_columns) < 2:

                st.warning(
                    "At least two numeric columns are required."
                )

            else:

                correlation = df[
                    numeric_columns
                ].corr()

                fig = px.imshow(
                    correlation,
                    text_auto=True,
                    title="Correlation Heatmap"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# =========================================================
# TRAIN MODEL
# =========================================================

elif page == "🤖 Train Model":

    st.header("🤖 Train Machine Learning Model")

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df.copy()

        # Target
        target_column = st.selectbox(
            "🎯 Select Target Column",
            df.columns
        )

        detected_task = detect_task(
            df[target_column]
        )

        st.info(
            f"Automatically detected task: **{detected_task}**"
        )

        task = st.radio(
            "Machine Learning Task",
            [
                "Classification",
                "Regression"
            ],
            index=(
                0
                if detected_task == "Classification"
                else 1
            )
        )

        # Model selection
        if task == "Classification":

            model_name = st.selectbox(
                "Select Model",
                [
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest"
                ]
            )

        else:

            model_name = st.selectbox(
                "Select Model",
                [
                    "Linear Regression",
                    "Decision Tree",
                    "Random Forest"
                ]
            )

        # Test size
        test_size = st.slider(
            "Test Size",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05
        )

        if st.button(
            "🚀 Train Model",
            type="primary"
        ):

            try:

                # Features
                X = prepare_features(
                    df,
                    target_column
                )

                # Target
                y = df[target_column].copy()

                # Remove missing targets
                valid = ~y.isna()

                X = X.loc[valid]
                y = y.loc[valid]

                if X.shape[1] == 0:

                    st.error(
                        "No usable feature columns found."
                    )

                    st.stop()

                # -------------------------------------------------
                # CLASSIFICATION
                # -------------------------------------------------

                if task == "Classification":

                    label_encoder = LabelEncoder()

                    y_encoded = label_encoder.fit_transform(
                        y.astype(str)
                    )

                    unique_classes, counts = np.unique(
                        y_encoded,
                        return_counts=True
                    )

                    if (
                        len(unique_classes) > 1
                        and counts.min() >= 2
                    ):

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y_encoded,
                            test_size=test_size,
                            random_state=42,
                            stratify=y_encoded
                        )

                    else:

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y_encoded,
                            test_size=test_size,
                            random_state=42
                        )

                    model = create_model(
                        task,
                        model_name
                    )

                    model.fit(
                        X_train,
                        y_train
                    )

                    predictions = model.predict(
                        X_test
                    )

                    accuracy = accuracy_score(
                        y_test,
                        predictions
                    )

                    st.session_state.model = model
                    st.session_state.feature_columns = X.columns.tolist()
                    st.session_state.target_column = target_column
                    st.session_state.task = task
                    st.session_state.label_encoder = label_encoder
                    st.session_state.model_name = model_name

                    save_model()

                    st.success(
                        "✅ Model trained successfully!"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Accuracy",
                            f"{accuracy * 100:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Training Rows",
                            len(X_train)
                        )

                    # Classification report
                    st.subheader(
                        "📄 Classification Report"
                    )

                    report = classification_report(
                        y_test,
                        predictions,
                        target_names=label_encoder.classes_,
                        output_dict=True,
                        zero_division=0
                    )

                    report_df = pd.DataFrame(
                        report
                    ).transpose()

                    st.dataframe(
                        report_df,
                        use_container_width=True
                    )

                    # Actual vs predicted
                    actual = label_encoder.inverse_transform(
                        y_test
                    )

                    predicted = label_encoder.inverse_transform(
                        predictions
                    )

                    results_df = pd.DataFrame({
                        "Actual": actual,
                        "Predicted": predicted
                    })

                    st.subheader(
                        "Actual vs Predicted"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                # -------------------------------------------------
                # REGRESSION
                # -------------------------------------------------

                else:

                    y = pd.to_numeric(
                        y,
                        errors="coerce"
                    )

                    valid = ~y.isna()

                    X = X.loc[valid]
                    y = y.loc[valid]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        random_state=42
                    )

                    model = create_model(
                        task,
                        model_name
                    )

                    model.fit(
                        X_train,
                        y_train
                    )

                    predictions = model.predict(
                        X_test
                    )

                    mae = mean_absolute_error(
                        y_test,
                        predictions
                    )

                    mse = mean_squared_error(
                        y_test,
                        predictions
                    )

                    rmse = np.sqrt(
                        mse
                    )

                    r2 = r2_score(
                        y_test,
                        predictions
                    )

                    st.session_state.model = model
                    st.session_state.feature_columns = X.columns.tolist()
                    st.session_state.target_column = target_column
                    st.session_state.task = task
                    st.session_state.label_encoder = None
                    st.session_state.model_name = model_name

                    save_model()

                    st.success(
                        "✅ Regression model trained successfully!"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "MAE",
                            f"{mae:.4f}"
                        )

                    with col2:
                        st.metric(
                            "MSE",
                            f"{mse:.4f}"
                        )

                    with col3:
                        st.metric(
                            "RMSE",
                            f"{rmse:.4f}"
                        )

                    with col4:
                        st.metric(
                            "R² Score",
                            f"{r2:.4f}"
                        )

                    results_df = pd.DataFrame({
                        "Actual": y_test.values,
                        "Predicted": predictions
                    })

                    st.subheader(
                        "Actual vs Predicted"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.header("🔮 Make Prediction")

    if st.session_state.model is None:

        st.warning(
            "Please train a model first."
        )

    else:

        model = st.session_state.model
        feature_columns = st.session_state.feature_columns
        target_column = st.session_state.target_column
        task = st.session_state.task

        st.success(
            f"Model: **{st.session_state.model_name}**"
        )

        st.write(
            f"Target: **{target_column}**"
        )

        st.subheader(
            "Enter Feature Values"
        )

        df = st.session_state.df

        input_data = {}

        original_features = [
            column
            for column in df.columns
            if column != target_column
        ]

        for column in original_features:

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                values = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:
                    default_value = float(
                        values.median()
                    )
                else:
                    default_value = 0.0

                input_data[column] = st.number_input(
                    column,
                    value=default_value
                )

            else:

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(values) > 0:

                    input_data[column] = st.selectbox(
                        column,
                        values
                    )

                else:

                    input_data[column] = st.text_input(
                        column
                    )

        if st.button(
            "🔮 Predict",
            type="primary"
        ):

            try:

                input_df = pd.DataFrame(
                    [input_data]
                )

                input_encoded = pd.get_dummies(
                    input_df,
                    drop_first=False
                )

                input_encoded = input_encoded.reindex(
                    columns=feature_columns,
                    fill_value=0
                )

                input_encoded = input_encoded.astype(
                    float
                )

                prediction = model.predict(
                    input_encoded
                )

                if task == "Classification":

                    encoder = st.session_state.label_encoder

                    if encoder is not None:

                        result = encoder.inverse_transform(
                            prediction.astype(int)
                        )[0]

                    else:

                        result = prediction[0]

                    st.success(
                        f"🎯 Prediction: **{result}**"
                    )

                else:

                    result = float(
                        prediction[0]
                    )

                    st.success(
                        f"🎯 Predicted Value: **{result:.4f}**"
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "ML Data Analysis Dashboard"
)

st.sidebar.caption(
    "Built with Python • Streamlit • Pandas • Scikit-learn • Plotly"
)

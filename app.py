import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ML Data App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border: 1px solid #81c784;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e3f2fd;
        border: 1px solid #64b5f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "model_pipeline" not in st.session_state:
    st.session_state.model_pipeline = None

if "target_column" not in st.session_state:
    st.session_state.target_column = None

if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = []

if "task_type" not in st.session_state:
    st.session_state.task_type = None

if "model_name" not in st.session_state:
    st.session_state.model_name = None

if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None

if "training_results" not in st.session_state:
    st.session_state.training_results = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def detect_task(y):
    """
    Automatically detect whether the problem is
    classification or regression.
    """

    if y.dtype == "object":
        return "Classification"

    if pd.api.types.is_bool_dtype(y):
        return "Classification"

    if pd.api.types.is_integer_dtype(y):
        unique_values = y.nunique()

        if unique_values <= 20:
            return "Classification"

        return "Regression"

    if pd.api.types.is_float_dtype(y):
        unique_values = y.nunique()

        if unique_values <= 10:
            return "Classification"

        return "Regression"

    return "Classification"


def prepare_target(y):
    """
    Prepare target column for machine learning.
    """

    encoder = None

    if y.dtype == "object" or pd.api.types.is_bool_dtype(y):
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y.astype(str))

        return y_encoded, encoder

    return y, encoder


def create_preprocessor(X):
    """
    Create preprocessing pipeline for numerical
    and categorical features.
    """

    numerical_columns = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


def get_model(task_type, model_name):
    """
    Return selected machine learning model.
    """

    if task_type == "Classification":

        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=1000
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


def dataframe_download(df):
    """
    Convert dataframe to CSV bytes.
    """

    return df.to_csv(index=False).encode("utf-8")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🤖 ML Data App")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📂 Upload Dataset",
        "📊 Data Analysis",
        "🤖 Train Model",
        "🔮 Prediction",
    ],
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **ML Data App**

    Upload your dataset, analyze the data,
    train a machine learning model and
    make predictions.
    """
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🤖 ML Data App</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        "Machine Learning Data Analysis & Prediction Dashboard"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.session_state.df is None:

        st.info(
            "No dataset uploaded yet. "
            "Go to **Upload Dataset** to get started."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Dataset",
                "Not Loaded",
            )

        with col2:
            st.metric(
                "Rows",
                "0",
            )

        with col3:
            st.metric(
                "Columns",
                "0",
            )

        with col4:
            st.metric(
                "Model",
                "Not Trained",
            )

    else:

        df = st.session_state.df

        st.success(
            "Dataset loaded successfully."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                df.shape[0],
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1],
            )

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum()),
            )

        with col4:

            if st.session_state.model is not None:
                st.metric(
                    "Model",
                    st.session_state.model_name,
                )
            else:
                st.metric(
                    "Model",
                    "Not Trained",
                )

        st.markdown("---")

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

        st.subheader("📌 Dataset Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Rows:**", df.shape[0])
            st.write("**Columns:**", df.shape[1])

        with col2:
            st.write(
                "**Duplicate Rows:**",
                int(df.duplicated().sum()),
            )
            st.write(
                "**Missing Values:**",
                int(df.isnull().sum().sum()),
            )


# =========================================================
# UPLOAD DATASET
# =========================================================

elif page == "📂 Upload Dataset":

    st.title("📂 Upload Dataset")

    st.write(
        "Upload a CSV file to start your machine learning workflow."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.session_state.df = df

            # Reset model when new dataset is uploaded
            st.session_state.model = None
            st.session_state.model_pipeline = None
            st.session_state.target_column = None
            st.session_state.feature_columns = []
            st.session_state.task_type = None
            st.session_state.model_name = None
            st.session_state.label_encoder = None
            st.session_state.training_results = None

            st.success(
                "Dataset uploaded successfully!"
            )

            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Rows",
                    df.shape[0],
                )

            with col2:
                st.metric(
                    "Columns",
                    df.shape[1],
                )

            with col3:
                st.metric(
                    "Missing Values",
                    int(df.isnull().sum().sum()),
                )

            st.subheader(
                "📋 Dataset Preview"
            )

            st.dataframe(
                df.head(20),
                use_container_width=True,
            )

            st.download_button(
                label="⬇️ Download Uploaded Dataset",
                data=dataframe_download(df),
                file_name="uploaded_dataset.csv",
                mime="text/csv",
            )

        except Exception as e:

            st.error(
                f"Could not read the CSV file: {e}"
            )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis")

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📋 Overview",
                "🔎 Statistics",
                "📈 Visualizations",
                "🧹 Data Quality",
            ]
        )

        # -------------------------------------------------
        # OVERVIEW
        # -------------------------------------------------

        with tab1:

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                df,
                use_container_width=True,
            )

            st.subheader(
                "Column Information"
            )

            column_info = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Data Type": [
                        str(dtype)
                        for dtype in df.dtypes
                    ],
                    "Non-Null": [
                        df[col].notna().sum()
                        for col in df.columns
                    ],
                    "Unique Values": [
                        df[col].nunique()
                        for col in df.columns
                    ],
                    "Missing": [
                        df[col].isnull().sum()
                        for col in df.columns
                    ],
                }
            )

            st.dataframe(
                column_info,
                use_container_width=True,
            )

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        with tab2:

            st.subheader(
                "Descriptive Statistics"
            )

            st.dataframe(
                df.describe(
                    include="all"
                ).transpose(),
                use_container_width=True,
            )

        # -------------------------------------------------
        # VISUALIZATIONS
        # -------------------------------------------------

        with tab3:

            st.subheader(
                "📈 Data Visualization"
            )

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_columns = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            chart_type = st.selectbox(
                "Select Chart",
                [
                    "Histogram",
                    "Bar Chart",
                    "Scatter Plot",
                    "Line Chart",
                    "Pie Chart",
                    "Correlation Heatmap",
                ],
            )

            # Histogram
            if chart_type == "Histogram":

                if not numeric_columns:

                    st.warning(
                        "No numeric columns available."
                    )

                else:

                    column = st.selectbox(
                        "Select numeric column",
                        numeric_columns,
                    )

                    fig = px.histogram(
                        df,
                        x=column,
                        title=f"Distribution of {column}",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

            # Bar chart
            elif chart_type == "Bar Chart":

                if not categorical_columns:

                    st.warning(
                        "No categorical columns available."
                    )

                else:

                    column = st.selectbox(
                        "Select categorical column",
                        categorical_columns,
                    )

                    counts = (
                        df[column]
                        .astype(str)
                        .value_counts()
                        .reset_index()
                    )

                    counts.columns = [
                        "Category",
                        "Count",
                    ]

                    fig = px.bar(
                        counts,
                        x="Category",
                        y="Count",
                        title=f"Distribution of {column}",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

            # Scatter plot
            elif chart_type == "Scatter Plot":

                if len(numeric_columns) < 2:

                    st.warning(
                        "At least two numeric columns are required."
                    )

                else:

                    x_column = st.selectbox(
                        "X Axis",
                        numeric_columns,
                    )

                    y_column = st.selectbox(
                        "Y Axis",
                        numeric_columns,
                        index=1
                        if len(numeric_columns) > 1
                        else 0,
                    )

                    fig = px.scatter(
                        df,
                        x=x_column,
                        y=y_column,
                        title=f"{x_column} vs {y_column}",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

            # Line chart
            elif chart_type == "Line Chart":

                if len(numeric_columns) < 2:

                    st.warning(
                        "At least two numeric columns are required."
                    )

                else:

                    x_column = st.selectbox(
                        "X Axis",
                        numeric_columns,
                    )

                    y_column = st.selectbox(
                        "Y Axis",
                        numeric_columns,
                        index=1
                        if len(numeric_columns) > 1
                        else 0,
                    )

                    fig = px.line(
                        df,
                        x=x_column,
                        y=y_column,
                        title=f"{y_column} over {x_column}",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

            # Pie chart
            elif chart_type == "Pie Chart":

                if not categorical_columns:

                    st.warning(
                        "No categorical columns available."
                    )

                else:

                    column = st.selectbox(
                        "Select categorical column",
                        categorical_columns,
                    )

                    counts = (
                        df[column]
                        .astype(str)
                        .value_counts()
                        .reset_index()
                    )

                    counts.columns = [
                        "Category",
                        "Count",
                    ]

                    fig = px.pie(
                        counts,
                        names="Category",
                        values="Count",
                        title=f"{column} Distribution",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
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

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=correlation.values,
                            x=correlation.columns,
                            y=correlation.columns,
                            colorscale="Viridis",
                            zmin=-1,
                            zmax=1,
                        )
                    )

                    fig.update_layout(
                        title="Correlation Heatmap"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

        # -------------------------------------------------
        # DATA QUALITY
        # -------------------------------------------------

        with tab4:

            st.subheader(
                "🧹 Missing Values"
            )

            missing_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Missing Values": [
                        df[col].isnull().sum()
                        for col in df.columns
                    ],
                    "Missing Percentage": [
                        round(
                            df[col].isnull().mean() * 100,
                            2,
                        )
                        for col in df.columns
                    ],
                }
            )

            st.dataframe(
                missing_df,
                use_container_width=True,
            )

            st.subheader(
                "Duplicate Rows"
            )

            duplicate_count = int(
                df.duplicated().sum()
            )

            if duplicate_count == 0:

                st.success(
                    "No duplicate rows found."
                )

            else:

                st.warning(
                    f"{duplicate_count} duplicate rows found."
                )


# =========================================================
# TRAIN MODEL
# =========================================================

elif page == "🤖 Train Model":

    st.title("🤖 Train Machine Learning Model")

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df.copy()

        st.subheader(
            "1️⃣ Select Target Column"
        )

        target_column = st.selectbox(
            "Target column",
            df.columns,
        )

        X = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        if X.shape[1] == 0:

            st.error(
                "Your dataset must have at least one feature column."
            )

        elif y.nunique() < 2:

            st.error(
                "Target column must contain at least two different values."
            )

        else:

            detected_task = detect_task(y)

            st.info(
                f"Automatically detected task: **{detected_task}**"
            )

            task_type = st.selectbox(
                "Machine Learning Task",
                [
                    "Classification",
                    "Regression",
                ],
                index=0
                if detected_task == "Classification"
                else 1,
            )

            if task_type == "Classification":

                model_options = [
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest",
                ]

            else:

                model_options = [
                    "Linear Regression",
                    "Decision Tree",
                    "Random Forest",
                ]

            model_name = st.selectbox(
                "Select Model",
                model_options,
            )

            test_size = st.slider(
                "Test Size",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
            )

            random_state = st.number_input(
                "Random State",
                min_value=1,
                max_value=1000,
                value=42,
                step=1,
            )

            st.markdown("---")

            st.subheader(
                "2️⃣ Dataset Summary"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Samples",
                    len(df),
                )

            with col2:
                st.metric(
                    "Features",
                    X.shape[1],
                )

            with col3:
                st.metric(
                    "Target Classes",
                    y.nunique(),
                )

            st.markdown("---")

            train_button = st.button(
                "🚀 Train Model",
                type="primary",
                use_container_width=True,
            )

            if train_button:

                try:

                    # Prepare target
                    y_prepared, label_encoder = prepare_target(
                        y
                    )

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y_prepared,
                        test_size=test_size,
                        random_state=int(random_state),
                        stratify=y_prepared
                        if task_type == "Classification"
                        and len(np.unique(y_prepared)) > 1
                        else None,
                    )

                    # Preprocessor
                    preprocessor = create_preprocessor(
                        X_train
                    )

                    # Model
                    model = get_model(
                        task_type,
                        model_name,
                    )

                    if model is None:

                        st.error(
                            "Unable to create selected model."
                        )

                    else:

                        pipeline = Pipeline(
                            steps=[
                                (
                                    "preprocessor",
                                    preprocessor,
                                ),
                                (
                                    "model",
                                    model,
                                ),
                            ]
                        )

                        with st.spinner(
                            "Training model..."
                        ):

                            pipeline.fit(
                                X_train,
                                y_train,
                            )

                        # Predictions
                        y_pred = pipeline.predict(
                            X_test
                        )

                        st.session_state.model = model
                        st.session_state.model_pipeline = pipeline
                        st.session_state.target_column = target_column
                        st.session_state.feature_columns = X.columns.tolist()
                        st.session_state.task_type = task_type
                        st.session_state.model_name = model_name
                        st.session_state.label_encoder = label_encoder

                        st.session_state.training_results = {
                            "X_test": X_test,
                            "y_test": y_test,
                            "y_pred": y_pred,
                        }

                        st.success(
                            f"{model_name} trained successfully!"
                        )

                        # -------------------------------------------------
                        # CLASSIFICATION METRICS
                        # -------------------------------------------------

                        if task_type == "Classification":

                            accuracy = accuracy_score(
                                y_test,
                                y_pred,
                            )

                            st.subheader(
                                "📊 Classification Results"
                            )

                            col1, col2 = st.columns(2)

                            with col1:

                                st.metric(
                                    "Accuracy",
                                    f"{accuracy * 100:.2f}%",
                                )

                            with col2:

                                st.metric(
                                    "Test Samples",
                                    len(y_test),
                                )

                            st.subheader(
                                "Classification Report"
                            )

                            report = classification_report(
                                y_test,
                                y_pred,
                                output_dict=True,
                                zero_division=0,
                            )

                            report_df = pd.DataFrame(
                                report
                            ).transpose()

                            st.dataframe(
                                report_df,
                                use_container_width=True,
                            )

                            st.subheader(
                                "Confusion Matrix"
                            )

                            cm = confusion_matrix(
                                y_test,
                                y_pred,
                            )

                            fig = go.Figure(
                                data=go.Heatmap(
                                    z=cm,
                                    colorscale="Blues",
                                    text=cm,
                                    texttemplate="%{text}",
                                )
                            )

                            fig.update_layout(
                                xaxis_title="Predicted",
                                yaxis_title="Actual",
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

                        # -------------------------------------------------
                        # REGRESSION METRICS
                        # -------------------------------------------------

                        else:

                            mae = mean_absolute_error(
                                y_test,
                                y_pred,
                            )

                            mse = mean_squared_error(
                                y_test,
                                y_pred,
                            )

                            rmse = np.sqrt(
                                mse
                            )

                            r2 = r2_score(
                                y_test,
                                y_pred,
                            )

                            st.subheader(
                                "📊 Regression Results"
                            )

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "MAE",
                                    f"{mae:.4f}",
                                )

                            with col2:
                                st.metric(
                                    "MSE",
                                    f"{mse:.4f}",
                                )

                            with col3:
                                st.metric(
                                    "RMSE",
                                    f"{rmse:.4f}",
                                )

                            with col4:
                                st.metric(
                                    "R² Score",
                                    f"{r2:.4f}",
                                )

                            results_df = pd.DataFrame(
                                {
                                    "Actual": y_test,
                                    "Predicted": y_pred,
                                }
                            )

                            st.subheader(
                                "Actual vs Predicted"
                            )

                            st.dataframe(
                                results_df.head(20),
                                use_container_width=True,
                            )

                            fig = px.scatter(
                                results_df,
                                x="Actual",
                                y="Predicted",
                                title="Actual vs Predicted Values",
                            )

                            min_value = min(
                                results_df["Actual"].min(),
                                results_df["Predicted"].min(),
                            )

                            max_value = max(
                                results_df["Actual"].max(),
                                results_df["Predicted"].max(),
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=[
                                        min_value,
                                        max_value,
                                    ],
                                    y=[
                                        min_value,
                                        max_value,
                                    ],
                                    mode="lines",
                                    name="Perfect Prediction",
                                )
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

                        # -------------------------------------------------
                        # SAVE MODEL
                        # -------------------------------------------------

                        os.makedirs(
                            "models",
                            exist_ok=True,
                        )

                        model_data = {
                            "pipeline": pipeline,
                            "target_column": target_column,
                            "feature_columns": X.columns.tolist(),
                            "task_type": task_type,
                            "model_name": model_name,
                            "label_encoder": label_encoder,
                        }

                        model_path = os.path.join(
                            "models",
                            "model.pkl",
                        )

                        joblib.dump(
                            model_data,
                            model_path,
                        )

                        st.success(
                            "Model saved successfully."
                        )

                        # Download model
                        model_buffer = io.BytesIO()

                        joblib.dump(
                            model_data,
                            model_buffer,
                        )

                        model_buffer.seek(0)

                        st.download_button(
                            label="⬇️ Download Trained Model",
                            data=model_buffer,
                            file_name="model.pkl",
                            mime="application/octet-stream",
                        )

                except Exception as e:

                    st.error(
                        f"Model training failed: {e}"
                    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.title("🔮 Make Prediction")

    if st.session_state.model_pipeline is None:

        st.warning(
            "Please train a model first."
        )

    else:

        pipeline = st.session_state.model_pipeline

        target_column = st.session_state.target_column

        feature_columns = st.session_state.feature_columns

        task_type = st.session_state.task_type

        model_name = st.session_state.model_name

        df = st.session_state.df

        st.success(
            f"Using model: **{model_name}**"
        )

        st.write(
            f"Task: **{task_type}**"
        )

        st.write(
            f"Target: **{target_column}**"
        )

        st.markdown("---")

        st.subheader(
            "Enter Feature Values"
        )

        input_data = {}

        input_columns = st.columns(2)

        for index, column in enumerate(
            feature_columns
        ):

            column_data = df[column]

            with input_columns[index % 2]:

                if (
                    pd.api.types.is_numeric_dtype(
                        column_data
                    )
                ):

                    median_value = column_data.median()

                    if pd.isna(median_value):
                        median_value = 0

                    input_data[column] = st.number_input(
                        column,
                        value=float(median_value),
                    )

                else:

                    values = (
                        column_data
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if values:

                        input_data[column] = st.selectbox(
                            column,
                            values,
                        )

                    else:

                        input_data[column] = st.text_input(
                            column
                        )

        st.markdown("---")

        predict_button = st.button(
            "🔮 Predict",
            type="primary",
            use_container_width=True,
        )

        if predict_button:

            try:

                input_df = pd.DataFrame(
                    [input_data]
                )

                prediction = pipeline.predict(
                    input_df
                )

                result = prediction[0]

                # Decode classification result
                if (
                    task_type == "Classification"
                    and st.session_state.label_encoder
                    is not None
                ):

                    encoder = (
                        st.session_state.label_encoder
                    )

                    try:

                        result = encoder.inverse_transform(
                            [int(result)]
                        )[0]

                    except Exception:

                        result = str(result)

                st.success(
                    f"Prediction: **{result}**"
                )

                st.subheader(
                    "Prediction Details"
                )

                result_df = pd.DataFrame(
                    {
                        "Feature": list(
                            input_data.keys()
                        ),
                        "Value": list(
                            input_data.values()
                        ),
                    }
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🤖 ML Data App | Built with Python, "
    "Streamlit, Pandas, Scikit-learn and Plotly"
)

import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ML Data Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #f6f8fc;
    }

    /* Remove default top spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    /* Sidebar radio */
    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        background: transparent;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 4px;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label:hover {
        background: #1f2937;
    }

    /* Main heading */
    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 3px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    /* Cards */
    .dashboard-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
        min-height: 120px;
    }

    .card-label {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .card-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
    }

    .card-icon {
        font-size: 25px;
        margin-bottom: 8px;
    }

    /* Section */
    .section-title {
        color: #111827;
        font-size: 22px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Info cards */
    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .info-card h3 {
        color: #111827;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #6b7280;
        line-height: 1.6;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 12px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        padding: 20px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "df": None,
    "model": None,
    "model_pipeline": None,
    "target_column": None,
    "feature_columns": [],
    "task_type": None,
    "model_name": None,
    "label_encoder": None,
    "training_results": None,
    "accuracy": None,
    "metrics": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_task(y):
    if y.dtype == "object":
        return "Classification"

    if pd.api.types.is_bool_dtype(y):
        return "Classification"

    if pd.api.types.is_integer_dtype(y):
        if y.nunique() <= 20:
            return "Classification"
        return "Regression"

    if pd.api.types.is_float_dtype(y):
        if y.nunique() <= 10:
            return "Classification"
        return "Regression"

    return "Classification"


def prepare_target(y):
    encoder = None

    if (
        y.dtype == "object"
        or pd.api.types.is_bool_dtype(y)
    ):
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(
            y.astype(str)
        )
        return y_encoded, encoder

    return y, encoder


def create_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        [
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
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
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

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
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


def get_model(task, model_name):

    if task == "Classification":

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
                n_estimators=150,
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
                n_estimators=150,
                random_state=42
            )

    return None


def reset_model():

    st.session_state.model = None
    st.session_state.model_pipeline = None
    st.session_state.target_column = None
    st.session_state.feature_columns = []
    st.session_state.task_type = None
    st.session_state.model_name = None
    st.session_state.label_encoder = None
    st.session_state.training_results = None
    st.session_state.accuracy = None
    st.session_state.metrics = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:10px 5px 25px 5px;">
            <div style="font-size:32px;">🤖</div>
            <div style="
                font-size:22px;
                font-weight:800;
                color:white;
                margin-top:5px;">
                ML Data App
            </div>
            <div style="
                font-size:12px;
                color:#9ca3af;
                margin-top:4px;">
                Data Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "📂 Upload Dataset",
            "📊 Data Analysis",
            "🤖 Train Model",
            "🔮 Prediction",
        ],
    )

    st.markdown("---")

    if st.session_state.df is not None:

        st.success("Dataset Loaded")

        st.caption(
            f"Rows: {len(st.session_state.df)}"
        )

        st.caption(
            f"Columns: {len(st.session_state.df.columns)}"
        )

    else:

        st.warning("No Dataset")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="hero-title">'
        "Machine Learning Dashboard"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Analyze datasets, visualize insights, train ML models "
        "and generate predictions from one platform."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    if st.session_state.df is not None:

        df = st.session_state.df

        total_rows = len(df)
        total_columns = len(df.columns)
        missing_values = int(
            df.isnull().sum().sum()
        )

        if st.session_state.model is not None:
            model_status = "Trained"
        else:
            model_status = "Not Trained"

    else:

        total_rows = 0
        total_columns = 0
        missing_values = 0
        model_status = "Not Trained"

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-icon">📄</div>
                <div class="card-label">Dataset Rows</div>
                <div class="card-value">
                    {total_rows:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-icon">📊</div>
                <div class="card-label">Features</div>
                <div class="card-value">
                    {total_columns}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-icon">⚠️</div>
                <div class="card-label">Missing Values</div>
                <div class="card-value">
                    {missing_values}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-icon">🤖</div>
                <div class="card-label">Model Status</div>
                <div class="card-value"
                     style="font-size:21px;">
                    {model_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">'
        "🚀 Machine Learning Workflow"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="info-card">
                <h3>📂 Upload Data</h3>
                <p>
                Upload your CSV dataset and instantly
                inspect rows, columns, data types and
                missing values.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Explore Data</h3>
                <p>
                Understand your data using statistics,
                charts, distributions, correlations
                and data-quality analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            """
            <div class="info-card">
                <h3>🤖 Build Model</h3>
                <p>
                Train classification or regression
                models and evaluate their performance
                using machine learning metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">'
        "📌 Platform Features"
        "</div>",
        unsafe_allow_html=True,
    )

    features = [
        "CSV Dataset Upload",
        "Data Preview",
        "Missing Value Analysis",
        "Statistical Analysis",
        "Interactive Visualizations",
        "Classification Models",
        "Regression Models",
        "Model Evaluation",
        "Prediction",
        "Model Download",
    ]

    cols = st.columns(5)

    for i, feature in enumerate(features):

        with cols[i % 5]:

            st.markdown(
                f"""
                <div style="
                    background:white;
                    border:1px solid #e5e7eb;
                    border-radius:10px;
                    padding:12px;
                    margin-bottom:10px;
                    font-size:13px;
                    font-weight:600;">
                    ✓ {feature}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# UPLOAD DATASET
# ============================================================

elif page == "📂 Upload Dataset":

    st.markdown(
        '<div class="hero-title">'
        "Upload Dataset"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Import your CSV dataset and prepare it for analysis."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your CSV file here",
        type=["csv"],
        help="Only CSV files are supported.",
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.session_state.df = df

            reset_model()

            st.success(
                "Dataset uploaded successfully!"
            )

            st.markdown(
                '<div class="section-title">'
                "Dataset Summary"
                "</div>",
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Rows",
                    f"{len(df):,}",
                )

            with c2:
                st.metric(
                    "Columns",
                    len(df.columns),
                )

            with c3:
                st.metric(
                    "Missing",
                    int(
                        df.isnull()
                        .sum()
                        .sum()
                    ),
                )

            with c4:
                st.metric(
                    "Duplicates",
                    int(
                        df.duplicated()
                        .sum()
                    ),
                )

            st.markdown(
                '<div class="section-title">'
                "Dataset Preview"
                "</div>",
                unsafe_allow_html=True,
            )

            st.dataframe(
                df.head(15),
                use_container_width=True,
                height=420,
            )

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Dataset",
                data=csv_data,
                file_name="dataset.csv",
                mime="text/csv",
            )

        except Exception as e:

            st.error(
                f"Unable to read dataset: {e}"
            )

    else:

        st.markdown(
            """
            <div style="
                background:white;
                border:1px dashed #94a3b8;
                border-radius:14px;
                padding:35px;
                text-align:center;
                margin-top:15px;">
                <div style="font-size:45px;">📁</div>
                <h3>Upload your CSV dataset</h3>
                <p style="color:#6b7280;">
                    Supported format: CSV
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.markdown(
        '<div class="hero-title">'
        "Data Analysis"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📋 Overview",
                "📈 Statistics",
                "📊 Visualizations",
                "🧹 Data Quality",
            ]
        )

        # ====================================================
        # OVERVIEW
        # ====================================================

        with tab1:

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                df,
                use_container_width=True,
                height=430,
            )

            st.subheader(
                "Column Information"
            )

            info_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Type": [
                        str(x)
                        for x in df.dtypes
                    ],
                    "Non-Null": [
                        df[c].notna().sum()
                        for c in df.columns
                    ],
                    "Missing": [
                        df[c].isna().sum()
                        for c in df.columns
                    ],
                    "Unique": [
                        df[c].nunique()
                        for c in df.columns
                    ],
                }
            )

            st.dataframe(
                info_df,
                use_container_width=True,
            )

        # ====================================================
        # STATISTICS
        # ====================================================

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

        # ====================================================
        # VISUALIZATIONS
        # ====================================================

        with tab3:

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_cols = df.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ]
            ).columns.tolist()

            chart_type = st.selectbox(
                "Choose visualization",
                [
                    "Histogram",
                    "Bar Chart",
                    "Scatter Plot",
                    "Line Chart",
                    "Pie Chart",
                    "Correlation Heatmap",
                ],
            )

            # ------------------------------------------------
            # HISTOGRAM
            # ------------------------------------------------

            if chart_type == "Histogram":

                if numeric_cols:

                    column = st.selectbox(
                        "Numeric column",
                        numeric_cols,
                    )

                    fig = px.histogram(
                        df,
                        x=column,
                        title=f"Distribution of {column}",
                        marginal="box",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "No numeric columns available."
                    )

            # ------------------------------------------------
            # BAR
            # ------------------------------------------------

            elif chart_type == "Bar Chart":

                if categorical_cols:

                    column = st.selectbox(
                        "Category column",
                        categorical_cols,
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
                        title=f"{column} Distribution",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "No categorical columns available."
                    )

            # ------------------------------------------------
            # SCATTER
            # ------------------------------------------------

            elif chart_type == "Scatter Plot":

                if len(numeric_cols) >= 2:

                    c1, c2 = st.columns(2)

                    with c1:

                        x_col = st.selectbox(
                            "X Axis",
                            numeric_cols,
                        )

                    with c2:

                        y_col = st.selectbox(
                            "Y Axis",
                            numeric_cols,
                            index=1,
                        )

                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{x_col} vs {y_col}",
                        trendline=None,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "At least two numeric columns are required."
                    )

            # ------------------------------------------------
            # LINE
            # ------------------------------------------------

            elif chart_type == "Line Chart":

                if len(numeric_cols) >= 2:

                    x_col = st.selectbox(
                        "X Axis",
                        numeric_cols,
                    )

                    y_col = st.selectbox(
                        "Y Axis",
                        numeric_cols,
                        index=1,
                    )

                    fig = px.line(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} over {x_col}",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "At least two numeric columns are required."
                    )

            # ------------------------------------------------
            # PIE
            # ------------------------------------------------

            elif chart_type == "Pie Chart":

                if categorical_cols:

                    column = st.selectbox(
                        "Category",
                        categorical_cols,
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
                        hole=0.35,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "No categorical columns available."
                    )

            # ------------------------------------------------
            # CORRELATION
            # ------------------------------------------------

            elif chart_type == "Correlation Heatmap":

                if len(numeric_cols) >= 2:

                    corr = df[
                        numeric_cols
                    ].corr()

                    fig = go.Figure(
                        data=[
                            go.Heatmap(
                                z=corr.values,
                                x=corr.columns,
                                y=corr.columns,
                                colorscale="Viridis",
                                zmin=-1,
                                zmax=1,
                            )
                        ]
                    )

                    fig.update_layout(
                        title="Feature Correlation",
                        height=600,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "At least two numeric columns are required."
                    )

        # ====================================================
        # DATA QUALITY
        # ====================================================

        with tab4:

            st.subheader(
                "Missing Values"
            )

            missing_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Missing": [
                        df[c].isna().sum()
                        for c in df.columns
                    ],
                    "Percentage": [
                        round(
                            df[c].isna().mean()
                            * 100,
                            2,
                        )
                        for c in df.columns
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

            duplicates = int(
                df.duplicated().sum()
            )

            if duplicates == 0:

                st.success(
                    "✓ No duplicate rows found."
                )

            else:

                st.warning(
                    f"{duplicates} duplicate rows found."
                )


# ============================================================
# TRAIN MODEL
# ============================================================

elif page == "🤖 Train Model":

    st.markdown(
        '<div class="hero-title">'
        "Train Machine Learning Model"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Configure, train and evaluate your machine learning model."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df.copy()

        st.markdown(
            '<div class="section-title">'
            "1. Model Configuration"
            "</div>",
            unsafe_allow_html=True,
        )

        target_column = st.selectbox(
            "Select Target Column",
            df.columns,
        )

        X = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        if X.shape[1] == 0:

            st.error(
                "At least one feature column is required."
            )

        elif y.nunique() < 2:

            st.error(
                "Target column must contain at least two unique values."
            )

        else:

            detected_task = detect_task(y)

            st.info(
                f"Detected problem type: **{detected_task}**"
            )

            task_type = st.selectbox(
                "Problem Type",
                [
                    "Classification",
                    "Regression",
                ],
                index=(
                    0
                    if detected_task == "Classification"
                    else 1
                ),
            )

            if task_type == "Classification":

                models = [
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest",
                ]

            else:

                models = [
                    "Linear Regression",
                    "Decision Tree",
                    "Random Forest",
                ]

            model_name = st.selectbox(
                "Choose Algorithm",
                models,
            )

            c1, c2 = st.columns(2)

            with c1:

                test_size = st.slider(
                    "Test Data Percentage",
                    10,
                    40,
                    20,
                )

            with c2:

                random_state = st.number_input(
                    "Random State",
                    min_value=1,
                    max_value=9999,
                    value=42,
                )

            st.markdown(
                '<div class="section-title">'
                "2. Dataset Information"
                "</div>",
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Samples",
                    f"{len(df):,}",
                )

            with c2:

                st.metric(
                    "Features",
                    len(X.columns),
                )

            with c3:

                st.metric(
                    "Target Values",
                    y.nunique(),
                )

            st.markdown("---")

            if st.button(
                "🚀 Train Model",
                type="primary",
                use_container_width=True,
            ):

                try:

                    y_prepared, encoder = prepare_target(
                        y
                    )

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y_prepared,
                            test_size=test_size / 100,
                            random_state=int(
                                random_state
                            ),
                            stratify=(
                                y_prepared
                                if task_type
                                == "Classification"
                                else None
                            ),
                        )
                    )

                    preprocessor = create_preprocessor(
                        X_train
                    )

                    model = get_model(
                        task_type,
                        model_name,
                    )

                    if model is None:

                        st.error(
                            "Model could not be created."
                        )

                    else:

                        pipeline = Pipeline(
                            [
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
                            "Training your model..."
                        ):

                            pipeline.fit(
                                X_train,
                                y_train,
                            )

                        predictions = pipeline.predict(
                            X_test
                        )

                        # Save state
                        st.session_state.model = model
                        st.session_state.model_pipeline = pipeline
                        st.session_state.target_column = target_column
                        st.session_state.feature_columns = X.columns.tolist()
                        st.session_state.task_type = task_type
                        st.session_state.model_name = model_name
                        st.session_state.label_encoder = encoder

                        st.session_state.training_results = {
                            "y_test": y_test,
                            "predictions": predictions,
                            "X_test": X_test,
                        }

                        st.success(
                            f"✓ {model_name} trained successfully!"
                        )

                        # =================================================
                        # CLASSIFICATION
                        # =================================================

                        if task_type == "Classification":

                            accuracy = accuracy_score(
                                y_test,
                                predictions,
                            )

                            st.session_state.accuracy = accuracy

                            st.markdown(
                                '<div class="section-title">'
                                "Model Performance"
                                "</div>",
                                unsafe_allow_html=True,
                            )

                            c1, c2, c3 = st.columns(3)

                            with c1:

                                st.metric(
                                    "Accuracy",
                                    f"{accuracy * 100:.2f}%",
                                )

                            with c2:

                                st.metric(
                                    "Training Samples",
                                    len(y_train),
                                )

                            with c3:

                                st.metric(
                                    "Testing Samples",
                                    len(y_test),
                                )

                            st.subheader(
                                "Classification Report"
                            )

                            report = classification_report(
                                y_test,
                                predictions,
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
                                predictions,
                            )

                            fig = px.imshow(
                                cm,
                                text_auto=True,
                                title="Confusion Matrix",
                                labels={
                                    "x": "Predicted",
                                    "y": "Actual",
                                },
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

                        # =================================================
                        # REGRESSION
                        # =================================================

                        else:

                            mae = mean_absolute_error(
                                y_test,
                                predictions,
                            )

                            mse = mean_squared_error(
                                y_test,
                                predictions,
                            )

                            rmse = np.sqrt(mse)

                            r2 = r2_score(
                                y_test,
                                predictions,
                            )

                            st.session_state.metrics = {
                                "MAE": mae,
                                "MSE": mse,
                                "RMSE": rmse,
                                "R2": r2,
                            }

                            st.markdown(
                                '<div class="section-title">'
                                "Model Performance"
                                "</div>",
                                unsafe_allow_html=True,
                            )

                            c1, c2, c3, c4 = st.columns(4)

                            with c1:
                                st.metric(
                                    "MAE",
                                    f"{mae:.3f}",
                                )

                            with c2:
                                st.metric(
                                    "MSE",
                                    f"{mse:.3f}",
                                )

                            with c3:
                                st.metric(
                                    "RMSE",
                                    f"{rmse:.3f}",
                                )

                            with c4:
                                st.metric(
                                    "R²",
                                    f"{r2:.3f}",
                                )

                            results = pd.DataFrame(
                                {
                                    "Actual": y_test,
                                    "Predicted": predictions,
                                }
                            )

                            st.subheader(
                                "Actual vs Predicted"
                            )

                            st.dataframe(
                                results.head(20),
                                use_container_width=True,
                            )

                            fig = px.scatter(
                                results,
                                x="Actual",
                                y="Predicted",
                                title="Actual vs Predicted",
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

                        # =================================================
                        # SAVE MODEL
                        # =================================================

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
                            "label_encoder": encoder,
                        }

                        model_path = os.path.join(
                            "models",
                            "model.pkl",
                        )

                        joblib.dump(
                            model_data,
                            model_path,
                        )

                        model_buffer = io.BytesIO()

                        joblib.dump(
                            model_data,
                            model_buffer,
                        )

                        model_buffer.seek(0)

                        st.download_button(
                            "⬇️ Download Trained Model",
                            data=model_buffer,
                            file_name="model.pkl",
                            mime="application/octet-stream",
                            use_container_width=True,
                        )

                except Exception as e:

                    st.error(
                        f"Training failed: {e}"
                    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.markdown(
        '<div class="hero-title">'
        "Make Prediction"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Enter feature values and generate a machine learning prediction."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.model_pipeline is None:

        st.warning(
            "No trained model found."
        )

        st.info(
            "Go to **Train Model** and train a model first."
        )

    else:

        pipeline = st.session_state.model_pipeline

        feature_columns = (
            st.session_state.feature_columns
        )

        task_type = st.session_state.task_type
        model_name = st.session_state.model_name
        target_column = st.session_state.target_column
        df = st.session_state.df

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Model",
                model_name,
            )

        with c2:

            st.metric(
                "Task",
                task_type,
            )

        with c3:

            st.metric(
                "Target",
                target_column,
            )

        st.markdown(
            '<div class="section-title">'
            "Enter Input Values"
            "</div>",
            unsafe_allow_html=True,
        )

        input_data = {}

        cols = st.columns(2)

        for i, column in enumerate(
            feature_columns
        ):

            series = df[column]

            with cols[i % 2]:

                if pd.api.types.is_numeric_dtype(
                    series
                ):

                    default = series.median()

                    if pd.isna(default):
                        default = 0

                    input_data[column] = st.number_input(
                        column,
                        value=float(default),
                    )

                else:

                    values = (
                        series
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

        if st.button(
            "🔮 Generate Prediction",
            type="primary",
            use_container_width=True,
        ):

            try:

                input_df = pd.DataFrame(
                    [input_data]
                )

                prediction = pipeline.predict(
                    input_df
                )

                result = prediction[0]

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

                st.markdown(
                    f"""
                    <div style="
                        background:white;
                        border:1px solid #d1d5db;
                        border-radius:16px;
                        padding:30px;
                        text-align:center;
                        margin-top:20px;">
                        <div style="
                            font-size:16px;
                            color:#6b7280;">
                            Prediction Result
                        </div>
                        <div style="
                            font-size:42px;
                            font-weight:800;
                            color:#111827;
                            margin-top:10px;">
                            {result}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-title">'
                    "Input Summary"
                    "</div>",
                    unsafe_allow_html=True,
                )

                input_display = pd.DataFrame(
                    {
                        "Feature": input_data.keys(),
                        "Value": input_data.values(),
                    }
                )

                st.dataframe(
                    input_display,
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🤖 ML Data Intelligence Platform
        <br>
        Built with Python • Streamlit • Pandas • Scikit-learn • Plotly
    </div>
    """,
    unsafe_allow_html=True,
)

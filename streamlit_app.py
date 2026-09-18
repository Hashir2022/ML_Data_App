import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ML Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ML Data Analysis & Prediction Dashboard")
st.caption("Upload your dataset, analyze it, train a machine learning model, and make predictions.")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

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


# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def detect_task(y):
    """
    Automatically detect classification or regression.
    """
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "Classification"

    if y.nunique() <= 10:
        return "Classification"

    return "Regression"


def prepare_features(df, target_column):
    """
    Convert categorical columns into numerical columns.
    """
    X = df.drop(columns=[target_column]).copy()

    # Remove completely empty columns
    X = X.dropna(axis=1, how="all")

    # Fill missing numerical values
    numeric_columns = X.select_dtypes(include=np.number).columns

    for col in numeric_columns:
        X[col] = X[col].fillna(X[col].median())

    # Fill missing categorical values
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for col in categorical_columns:
        if X[col].isna().any():
            mode = X[col].mode()

            if len(mode) > 0:
                X[col] = X[col].fillna(mode.iloc[0])
            else:
                X[col] = X[col].fillna("Unknown")

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=False)

    # Convert boolean to integer
    X = X.astype(float)

    return X


def create_model(task, model_name):
    """
    Create selected machine learning model.
    """

    if task == "Classification":

        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=2000)

        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(random_state=42)

        elif model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

    else:

        if model_name == "Linear Regression":
            return LinearRegression()

        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(random_state=42)

        elif model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

    return None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📂 Upload Dataset",
        "🔍 Data Analysis",
        "🤖 Train Model",
        "🔮 Prediction"
    ]
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    st.markdown("""
    ### Welcome to the ML Data Analysis Dashboard

    This application allows you to:

    - 📂 Upload CSV datasets
    - 🔍 Explore and analyze data
    - 📊 Create interactive visualizations
    - 🤖 Train machine learning models
    - 📈 Evaluate model performance
    - 🔮 Make predictions
    """)

    if st.session_state.df is not None:

        df = st.session_state.df

        st.success("Dataset loaded successfully!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

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

    else:

        st.info(
            "No dataset uploaded yet. Go to "
            "**Upload Dataset** from the sidebar."
        )


# ==================================================
# UPLOAD DATASET
# ==================================================

elif page == "📂 Upload Dataset":

    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.session_state.df = df

            st.session_state.model = None
            st.session_state.feature_columns = []
            st.session_state.target_column = None
            st.session_state.task = None
            st.session_state.label_encoder = None

            st.success("Dataset uploaded successfully!")

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rows", df.shape[0])

            with col2:
                st.metric("Columns", df.shape[1])

            with col3:
                st.metric(
                    "Missing Values",
                    int(df.isnull().sum().sum())
                )

        except Exception as e:

            st.error(
                f"Unable to read the CSV file: {e}"
            )


# ==================================================
# DATA ANALYSIS
# ==================================================

elif page == "🔍 Data Analysis":

    st.header("🔍 Data Analysis")

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        st.subheader("📋 Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Total Missing Values",
                int(df.isnull().sum().sum())
            )

        st.subheader("👀 Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.subheader("📊 Statistical Summary")

        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

        st.subheader("❓ Missing Values")

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values,
            "Missing Percentage": (
                df.isnull().mean().values * 100
            ).round(2)
        })

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        st.subheader("📈 Visualization")

        chart_type = st.selectbox(
            "Select Chart",
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

        if chart_type == "Histogram":

            if numeric_columns:

                column = st.selectbox(
                    "Select numeric column",
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

            else:

                st.warning(
                    "No numeric columns available."
                )

        elif chart_type == "Bar Chart":

            column = st.selectbox(
                "Select column",
                all_columns
            )

            value_counts = (
                df[column]
                .astype(str)
                .value_counts()
                .head(20)
                .reset_index()
            )

            value_counts.columns = [
                "Value",
                "Count"
            ]

            fig = px.bar(
                value_counts,
                x="Value",
                y="Count",
                title=f"Bar Chart - {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        elif chart_type == "Scatter Plot":

            if len(numeric_columns) >= 2:

                x_column = st.selectbox(
                    "X Axis",
                    numeric_columns,
                    key="scatter_x"
                )

                y_column = st.selectbox(
                    "Y Axis",
                    numeric_columns,
                    key="scatter_y"
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

            else:

                st.warning(
                    "At least two numeric columns are required."
                )

        elif chart_type == "Line Chart":

            if numeric_columns:

                column = st.selectbox(
                    "Select numeric column",
                    numeric_columns,
                    key="line_column"
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

            else:

                st.warning(
                    "No numeric columns available."
                )

        elif chart_type == "Pie Chart":

            column = st.selectbox(
                "Select column",
                all_columns,
                key="pie_column"
            )

            value_counts = (
                df[column]
                .astype(str)
                .value_counts()
                .head(10)
            )

            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Pie Chart - {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        elif chart_type == "Correlation Heatmap":

            if len(numeric_columns) >= 2:

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

            else:

                st.warning(
                    "At least two numeric columns are required."
                )


# ==================================================
# TRAIN MODEL
# ==================================================

elif page == "🤖 Train Model":

    st.header("🤖 Train Machine Learning Model")

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df.copy()

        st.subheader("1️⃣ Select Target Column")

        target_column = st.selectbox(
            "Target column",
            df.columns
        )

        detected_task = detect_task(
            df[target_column]
        )

        st.info(
            f"Automatically detected task: **{detected_task}**"
        )

        task = st.radio(
            "Select Task",
            ["Classification", "Regression"],
            index=0 if detected_task == "Classification" else 1
        )

        if task == "Classification":

            model_name = st.selectbox(
                "Select Classification Model",
                [
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest"
                ]
            )

        else:

            model_name = st.selectbox(
                "Select Regression Model",
                [
                    "Linear Regression",
                    "Decision Tree",
                    "Random Forest"
                ]
            )

        test_size = st.slider(
            "Test Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05
        )

        if st.button(
            "🚀 Train Model",
            type="primary"
        ):

            try:

                X = prepare_features(
                    df,
                    target_column
                )

                y = df[target_column].copy()

                # Remove rows where target is missing
                valid_rows = ~y.isna()

                X = X.loc[valid_rows]
                y = y.loc[valid_rows]

                if X.shape[1] == 0:

                    st.error(
                        "No usable feature columns were found."
                    )

                    st.stop()

                label_encoder = None

                # ------------------------------------------
                # CLASSIFICATION
                # ------------------------------------------

                if task == "Classification":

                    label_encoder = LabelEncoder()

                    y_encoded = label_encoder.fit_transform(
                        y.astype(str)
                    )

                    st.session_state.label_encoder = label_encoder

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y_encoded,
                        test_size=test_size,
                        random_state=42,
                        stratify=y_encoded
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

                    st.success(
                        "Model trained successfully!"
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

                # ------------------------------------------
                # REGRESSION
                # ------------------------------------------

                else:

                    y = pd.to_numeric(
                        y,
                        errors="coerce"
                    )

                    valid_rows = ~y.isna()

                    X = X.loc[valid_rows]
                    y = y.loc[valid_rows]

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

                    rmse = np.sqrt(mse)

                    r2 = r2_score(
                        y_test,
                        predictions
                    )

                    st.session_state.model = model
                    st.session_state.feature_columns = X.columns.tolist()
                    st.session_state.target_column = target_column
                    st.session_state.task = task
                    st.session_state.label_encoder = None

                    st.success(
                        "Regression model trained successfully!"
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

            except Exception as e:

                st.error(
                    f"Model training failed: {e}"
                )


# ==================================================
# PREDICTION
# ==================================================

elif page == "🔮 Prediction":

    st.header("🔮 Make Prediction")

    if st.session_state.model is None:

        st.warning(
            "Please train a model first."
        )

    else:

        df = st.session_state.df
        model = st.session_state.model
        feature_columns = st.session_state.feature_columns
        target_column = st.session_state.target_column
        task = st.session_state.task

        st.success(
            f"Model ready: **{type(model).__name__}**"
        )

        st.write(
            f"Target Column: **{target_column}**"
        )

        st.subheader("Enter Feature Values")

        input_data = {}

        original_features = [
            col for col in df.columns
            if col != target_column
        ]

        for col in original_features:

            if pd.api.types.is_numeric_dtype(
                df[col]
            ):

                default_value = float(
                    df[col].median()
                ) if not df[col].dropna().empty else 0.0

                input_data[col] = st.number_input(
                    col,
                    value=default_value
                )

            else:

                values = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if values:

                    input_data[col] = st.selectbox(
                        col,
                        values
                    )

                else:

                    input_data[col] = st.text_input(
                        col
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

                input_encoded = input_encoded.astype(float)

                prediction = model.predict(
                    input_encoded
                )

                if task == "Classification":

                    encoder = st.session_state.label_encoder

                    if encoder is not None:

                        predicted_class = encoder.inverse_transform(
                            prediction.astype(int)
                        )[0]

                    else:

                        predicted_class = prediction[0]

                    st.success(
                        f"🎯 Prediction: **{predicted_class}**"
                    )

                else:

                    predicted_value = float(
                        prediction[0]
                    )

                    st.success(
                        f"🎯 Predicted Value: **{predicted_value:.4f}**"
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.info(
    "ML Data Analysis Dashboard\n\n"
    "Built with Python, Streamlit & Scikit-learn."
)

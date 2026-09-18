from flask import Flask, render_template, request, redirect, url_for, flash
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


app = Flask(__name__)
app.secret_key = "ml-dashboard-secret-key"

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"

DATASET_PATH = os.path.join(UPLOAD_FOLDER, "dataset.csv")
MODEL_PATH = os.path.join(MODEL_FOLDER, "model.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None

    try:
        return pd.read_csv(DATASET_PATH)
    except Exception:
        return None


def chart_html(fig):
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn"
    )


def prepare_features(df):
    data = df.copy()

    # Missing values
    for col in data.columns:

        if data[col].dtype == "object":
            data[col] = data[col].fillna("Unknown")

        else:
            median = data[col].median()

            if pd.isna(median):
                median = 0

            data[col] = data[col].fillna(median)

    # Categorical columns -> numerical
    data = pd.get_dummies(
        data,
        drop_first=False
    )

    # Make sure everything is numeric
    for col in data.columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.fillna(0)

    return data


def detect_task(y):

    if (
        y.dtype == "object"
        or str(y.dtype) == "category"
        or str(y.dtype) == "bool"
    ):
        return "classification"

    if y.nunique() <= 10:
        return "classification"

    return "regression"


def create_model(model_name, task):

    if task == "classification":

        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=2000
            )

        if model_name == "Decision Tree":
            return DecisionTreeClassifier(
                random_state=42
            )

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

        return RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def index():

    df = load_dataset()

    if df is None:

        stats = {
            "rows": 0,
            "columns": 0,
            "missing": 0,
            "model": "Not Trained"
        }

    else:

        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing": int(df.isna().sum().sum()),
            "model": (
                "Trained"
                if os.path.exists(MODEL_PATH)
                else "Not Trained"
            )
        }

    return render_template(
        "index.html",
        stats=stats,
        dataset=df
    )


# =========================================================
# UPLOAD
# =========================================================

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("file")

    if not file or file.filename == "":
        flash("Please select a CSV file.")
        return redirect(url_for("index"))

    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files are supported.")
        return redirect(url_for("index"))

    try:

        df = pd.read_csv(file)

        if df.empty:
            flash("CSV file is empty.")
            return redirect(url_for("index"))

        df.to_csv(
            DATASET_PATH,
            index=False
        )

        # Remove old model
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        flash(
            f"Dataset uploaded successfully: "
            f"{len(df)} rows × {len(df.columns)} columns"
        )

    except Exception as e:

        flash(
            f"Upload error: {str(e)}"
        )

    return redirect(url_for("index"))


# =========================================================
# ANALYSIS
# =========================================================

@app.route("/analysis")
def analysis():

    df = load_dataset()

    if df is None:

        flash(
            "Please upload a CSV dataset first."
        )

        return redirect(url_for("index"))

    try:

        # -------------------------------------------------
        # Tables
        # -------------------------------------------------

        preview = df.head(10)

        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(df[col].dtype)
                for col in df.columns
            ],
            "Missing Values": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique Values": [
                int(df[col].nunique())
                for col in df.columns
            ]
        })

        missing = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": [
                int(df[col].isna().sum())
                for col in df.columns
            ]
        })

        describe = df.describe(
            include="all"
        ).transpose()

        describe = describe.fillna("")

        # -------------------------------------------------
        # Numeric / Categorical columns
        # -------------------------------------------------

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # =================================================
        # 1. HISTOGRAM
        # =================================================

        histogram_chart = None

        if numeric_columns:

            col = numeric_columns[0]

            fig = px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}",
                nbins=30
            )

            fig.update_layout(
                template="plotly_white",
                height=450
            )

            histogram_chart = chart_html(fig)

        # =================================================
        # 2. BAR CHART
        # =================================================

        bar_chart = None

        if categorical_columns:

            col = categorical_columns[0]

            counts = (
                df[col]
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )

            counts.columns = [
                col,
                "Count"
            ]

            fig = px.bar(
                counts,
                x=col,
                y="Count",
                title=f"Top Categories - {col}"
            )

            fig.update_layout(
                template="plotly_white",
                height=450
            )

            bar_chart = chart_html(fig)

        elif numeric_columns:

            col = numeric_columns[0]

            counts = (
                df[col]
                .value_counts()
                .head(10)
                .reset_index()
            )

            counts.columns = [
                col,
                "Count"
            ]

            fig = px.bar(
                counts,
                x=col,
                y="Count",
                title=f"Top Values - {col}"
            )

            fig.update_layout(
                template="plotly_white",
                height=450
            )

            bar_chart = chart_html(fig)

        # =================================================
        # 3. PIE CHART
        # =================================================

        pie_chart = None

        if categorical_columns:

            col = categorical_columns[0]

            counts = (
                df[col]
                .astype(str)
                .value_counts()
                .head(8)
                .reset_index()
            )

            counts.columns = [
                col,
                "Count"
            ]

            fig = px.pie(
                counts,
                names=col,
                values="Count",
                title=f"{col} Distribution"
            )

            fig.update_layout(
                template="plotly_white",
                height=450
            )

            pie_chart = chart_html(fig)

        # =================================================
        # 4. SCATTER PLOT
        # =================================================

        scatter_chart = None

        if len(numeric_columns) >= 2:

            x_col = numeric_columns[0]
            y_col = numeric_columns[1]

            scatter_df = df[
                [x_col, y_col]
            ].dropna()

            fig = px.scatter(
                scatter_df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}"
            )

            fig.update_layout(
                template="plotly_white",
                height=450
            )

            scatter_chart = chart_html(fig)

        # =================================================
        # 5. LINE CHART
        # =================================================

        line_chart = None

        if numeric_columns:

            col = numeric_columns[0]

            line_df = df[
                [col]
            ].dropna().head(100).copy()

            line_df["Row"] = range(
                1,
                len(line_df) + 1
            )

            fig = px.line(
                line_df,
                x="Row",
                y=col,
                title=f"{col} Trend"
            )

            fig.update_layout(
                template="plotly_white",
                height=450
            )

            line_chart = chart_html(fig)

        # =================================================
        # 6. CORRELATION HEATMAP
        # =================================================

        correlation_chart = None

        if len(numeric_columns) >= 2:

            corr = df[
                numeric_columns
            ].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap"
            )

            fig.update_layout(
                template="plotly_white",
                height=500
            )

            correlation_chart = chart_html(fig)

        return render_template(
            "analysis.html",

            preview=preview.to_html(
                classes="table table-striped table-hover",
                index=False
            ),

            column_info=column_info.to_html(
                classes="table table-striped table-hover",
                index=False
            ),

            missing=missing.to_html(
                classes="table table-striped table-hover",
                index=False
            ),

            describe=describe.to_html(
                classes="table table-striped table-hover"
            ),

            histogram_chart=histogram_chart,
            bar_chart=bar_chart,
            pie_chart=pie_chart,
            scatter_chart=scatter_chart,
            line_chart=line_chart,
            correlation_chart=correlation_chart,

            columns=df.columns.tolist()
        )

    except Exception as e:

        flash(
            f"Analysis error: {str(e)}"
        )

        return redirect(
            url_for("index")
        )


# =========================================================
# TRAIN MODEL
# =========================================================

@app.route("/train", methods=["POST"])
def train():

    df = load_dataset()

    if df is None:

        flash(
            "Please upload a dataset first."
        )

        return redirect(url_for("index"))

    target = request.form.get("target")
    model_name = request.form.get("model")

    if target not in df.columns:

        flash(
            "Invalid target column."
        )

        return redirect(url_for("analysis"))

    try:

        df = df.dropna(
            subset=[target]
        ).copy()

        X_original = df.drop(
            columns=[target]
        )

        y = df[target]

        task = detect_task(y)

        X = prepare_features(
            X_original
        )

        label_encoder = None

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        if task == "classification":

            label_encoder = LabelEncoder()

            y_final = label_encoder.fit_transform(
                y.astype(str)
            )

            if len(np.unique(y_final)) < 2:

                flash(
                    "Target must contain at least two classes."
                )

                return redirect(
                    url_for("analysis")
                )

        # -------------------------------------------------
        # Regression
        # -------------------------------------------------

        else:

            y_final = pd.to_numeric(
                y,
                errors="coerce"
            )

            valid = y_final.notna()

            X = X.loc[valid]
            y_final = y_final.loc[valid]

        # -------------------------------------------------
        # Split
        # -------------------------------------------------

        try:

            if task == "classification":

                class_counts = pd.Series(
                    y_final
                ).value_counts()

                if class_counts.min() >= 2:

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y_final,
                        test_size=0.2,
                        random_state=42,
                        stratify=y_final
                    )

                else:

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y_final,
                        test_size=0.2,
                        random_state=42
                    )

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y_final,
                    test_size=0.2,
                    random_state=42
                )

        except Exception:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_final,
                test_size=0.2,
                random_state=42
            )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model = create_model(
            model_name,
            task
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        metrics = {}

        if task == "classification":

            metrics["Accuracy"] = round(
                accuracy_score(
                    y_test,
                    predictions
                ) * 100,
                2
            )

            metrics["Precision"] = round(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            )

            metrics["Recall"] = round(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            )

            metrics["F1 Score"] = round(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            )

        else:

            mse = mean_squared_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(mse)

            metrics["MAE"] = round(
                mean_absolute_error(
                    y_test,
                    predictions
                ),
                4
            )

            metrics["RMSE"] = round(
                rmse,
                4
            )

            metrics["R² Score"] = round(
                r2_score(
                    y_test,
                    predictions
                ),
                4
            )

        # -------------------------------------------------
        # Save model
        # -------------------------------------------------

        model_data = {
            "model": model,
            "columns": X.columns.tolist(),
            "task": task,
            "target": target,
            "model_name": model_name,
            "label_encoder": label_encoder
        }

        joblib.dump(
            model_data,
            MODEL_PATH
        )

        # -------------------------------------------------
        # Performance Chart
        # -------------------------------------------------

        metric_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })

        fig = px.bar(
            metric_df,
            x="Metric",
            y="Value",
            text="Value",
            title="Model Performance"
        )

        fig.update_layout(
            template="plotly_white",
            height=450
        )

        ml_chart = chart_html(fig)

        return render_template(
            "result.html",
            metrics=metrics,
            task=task,
            target=target,
            model_name=model_name,
            ml_chart=ml_chart
        )

    except Exception as e:

        flash(
            f"Training error: {str(e)}"
        )

        return redirect(
            url_for("analysis")
        )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if not os.path.exists(MODEL_PATH):

        flash(
            "Please train a model first."
        )

        return redirect(
            url_for("index")
        )

    model_data = joblib.load(
        MODEL_PATH
    )

    model = model_data["model"]
    columns = model_data["columns"]
    task = model_data["task"]
    label_encoder = model_data["label_encoder"]

    prediction = None

    if request.method == "POST":

        try:

            values = {}

            for col in columns:

                value = request.form.get(
                    col,
                    ""
                )

                if value == "":

                    values[col] = np.nan

                else:

                    try:
                        values[col] = float(value)

                    except ValueError:
                        values[col] = value

            input_df = pd.DataFrame(
                [values]
            )

            input_df = prepare_features(
                input_df
            )

            input_df = input_df.reindex(
                columns=columns,
                fill_value=0
            )

            result = model.predict(
                input_df
            )[0]

            if task == "classification":

                result = label_encoder.inverse_transform(
                    [int(result)]
                )[0]

            else:

                result = round(
                    float(result),
                    4
                )

            prediction = result

        except Exception as e:

            flash(
                f"Prediction error: {str(e)}"
            )

    return render_template(
        "predict.html",
        columns=columns,
        prediction=prediction,
        task=task
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
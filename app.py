from flask import Flask, render_template, request, redirect, url_for, flash
import os
import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.io as pio

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "ml-data-app-secret-key"
)


# =========================================================
# FOLDERS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_dataset(filepath):
    """
    Load CSV dataset.
    """
    return pd.read_csv(filepath)


def chart_html(fig):
    """
    Convert Plotly figure to HTML.
    """
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn"
    )


def prepare_features(df, target_column):
    """
    Prepare features for machine learning.
    Handles missing values and categorical columns.
    """

    X = df.drop(
        columns=[target_column]
    ).copy()

    # Remove completely empty columns
    X = X.dropna(
        axis=1,
        how="all"
    )

    # Numerical columns
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

    # Convert boolean values
    X = X.astype(float)

    return X


def detect_task(y):
    """
    Automatically detect classification or regression.
    """

    if (
        y.dtype == "object"
        or str(y.dtype).startswith("category")
    ):
        return "Classification"

    if y.nunique() <= 10:
        return "Classification"

    return "Regression"


def create_model(task, model_name):
    """
    Create selected ML model.
    """

    if task == "Classification":

        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=2000
            )

        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(
                random_state=42
            )

        elif model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

    else:

        if model_name == "Linear Regression":
            return LinearRegression()

        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(
                random_state=42
            )

        elif model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

    return None


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "file" not in request.files:

        flash(
            "No file selected."
        )

        return redirect(
            url_for("index")
        )

    file = request.files["file"]

    if file.filename == "":

        flash(
            "Please select a CSV file."
        )

        return redirect(
            url_for("index")
        )

    if not file.filename.lower().endswith(".csv"):

        flash(
            "Only CSV files are supported."
        )

        return redirect(
            url_for("index")
        )

    try:

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        df = load_dataset(
            filepath
        )

        # Save current dataset information
        df.to_csv(
            filepath,
            index=False
        )

        flash(
            "Dataset uploaded successfully!"
        )

        return redirect(
            url_for(
                "analysis",
                filename=file.filename
            )
        )

    except Exception as e:

        flash(
            f"Error uploading dataset: {e}"
        )

        return redirect(
            url_for("index")
        )


# =========================================================
# ANALYSIS
# =========================================================

@app.route("/analysis")
def analysis():

    filename = request.args.get(
        "filename"
    )

    if not filename:

        flash(
            "No dataset selected."
        )

        return redirect(
            url_for("index")
        )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        flash(
            "Dataset not found."
        )

        return redirect(
            url_for("index")
        )

    try:

        df = load_dataset(
            filepath
        )

        # Basic information
        rows = len(df)
        columns = len(df.columns)

        missing_values = int(
            df.isnull().sum().sum()
        )

        duplicate_rows = int(
            df.duplicated().sum()
        )

        # Preview
        preview = df.head(10)

        # Dataset info
        info_data = []

        for column in df.columns:

            info_data.append({
                "Column": column,
                "Data Type": str(
                    df[column].dtype
                ),
                "Missing": int(
                    df[column].isnull().sum()
                ),
                "Unique": int(
                    df[column].nunique()
                )
            })

        info_df = pd.DataFrame(
            info_data
        )

        # Statistics
        try:

            description = df.describe(
                include="all"
            ).fillna("")

        except Exception:

            description = pd.DataFrame()

        return render_template(
            "analysis.html",
            filename=filename,
            data=preview.to_html(
                classes="table table-striped table-bordered",
                index=False
            ),
            info=info_df.to_html(
                classes="table table-striped table-bordered",
                index=False
            ),
            description=description.to_html(
                classes="table table-striped table-bordered"
            ),
            rows=rows,
            columns=columns,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows
        )

    except Exception as e:

        flash(
            f"Error analyzing dataset: {e}"
        )

        return redirect(
            url_for("index")
        )


# =========================================================
# TRAIN MODEL
# =========================================================

@app.route(
    "/train",
    methods=["POST"]
)
def train():

    filename = request.form.get(
        "filename"
    )

    target_column = request.form.get(
        "target_column"
    )

    model_name = request.form.get(
        "model"
    )

    test_size = request.form.get(
        "test_size",
        "0.2"
    )

    if not filename or not target_column:

        flash(
            "Dataset and target column are required."
        )

        return redirect(
            url_for("index")
        )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        flash(
            "Dataset not found."
        )

        return redirect(
            url_for("index")
        )

    try:

        df = load_dataset(
            filepath
        )

        if target_column not in df.columns:

            flash(
                "Target column not found."
            )

            return redirect(
                url_for(
                    "analysis",
                    filename=filename
                )
            )

        # Detect task
        task = detect_task(
            df[target_column]
        )

        # Prepare X
        X = prepare_features(
            df,
            target_column
        )

        # Target
        y = df[target_column].copy()

        # Remove missing target rows
        valid_rows = ~y.isna()

        X = X.loc[
            valid_rows
        ]

        y = y.loc[
            valid_rows
        ]

        # Convert target
        label_encoder = None

        if task == "Classification":

            label_encoder = LabelEncoder()

            y = label_encoder.fit_transform(
                y.astype(str)
            )

        else:

            y = pd.to_numeric(
                y,
                errors="coerce"
            )

            valid_target = ~pd.isna(y)

            X = X.loc[
                valid_target
            ]

            y = y.loc[
                valid_target
            ]

        # Test size
        try:

            test_size = float(
                test_size
            )

        except Exception:

            test_size = 0.2

        # Train/test split
        if task == "Classification":

            # Stratification only when possible
            unique_classes, class_counts = np.unique(
                y,
                return_counts=True
            )

            if (
                len(unique_classes) > 1
                and class_counts.min() >= 2
            ):

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y
                )

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42
                )

        else:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42
            )

        # Create model
        model = create_model(
            task,
            model_name
        )

        if model is None:

            flash(
                "Invalid model selected."
            )

            return redirect(
                url_for(
                    "analysis",
                    filename=filename
                )
            )

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Predict
        predictions = model.predict(
            X_test
        )

        # Save model package
        model_package = {
            "model": model,
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "task": task,
            "label_encoder": label_encoder
        }

        model_path = os.path.join(
            MODEL_FOLDER,
            "model.pkl"
        )

        joblib.dump(
            model_package,
            model_path
        )

        # Classification metrics
        if task == "Classification":

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            report = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            # Convert prediction labels
            if label_encoder is not None:

                try:

                    actual_labels = label_encoder.inverse_transform(
                        y_test.astype(int)
                    )

                    predicted_labels = label_encoder.inverse_transform(
                        predictions.astype(int)
                    )

                except Exception:

                    actual_labels = y_test
                    predicted_labels = predictions

            else:

                actual_labels = y_test
                predicted_labels = predictions

            result_df = pd.DataFrame({
                "Actual": actual_labels,
                "Predicted": predicted_labels
            })

            return render_template(
                "result.html",
                task=task,
                model=model_name,
                accuracy=round(
                    accuracy * 100,
                    2
                ),
                report=report_df.to_html(
                    classes="table table-striped table-bordered"
                ),
                results=result_df.to_html(
                    classes="table table-striped table-bordered",
                    index=False
                ),
                target_column=target_column,
                filename=filename
            )

        # Regression metrics
        else:

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

            result_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": predictions
            })

            return render_template(
                "result.html",
                task=task,
                model=model_name,
                mae=round(
                    mae,
                    4
                ),
                mse=round(
                    mse,
                    4
                ),
                rmse=round(
                    rmse,
                    4
                ),
                r2=round(
                    r2,
                    4
                ),
                results=result_df.to_html(
                    classes="table table-striped table-bordered",
                    index=False
                ),
                target_column=target_column,
                filename=filename
            )

    except Exception as e:

        flash(
            f"Model training failed: {e}"
        )

        return redirect(
            url_for(
                "analysis",
                filename=filename
            )
        )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    model_path = os.path.join(
        MODEL_FOLDER,
        "model.pkl"
    )

    if not os.path.exists(model_path):

        flash(
            "Please train a model first."
        )

        return redirect(
            url_for("index")
        )

    try:

        model_package = joblib.load(
            model_path
        )

        model = model_package["model"]
        feature_columns = model_package["feature_columns"]
        task = model_package["task"]
        target_column = model_package["target_column"]
        label_encoder = model_package.get(
            "label_encoder"
        )

        if request.method == "GET":

            return render_template(
                "predict.html",
                feature_columns=feature_columns,
                task=task,
                target_column=target_column
            )

        # POST prediction
        input_data = {}

        for column in feature_columns:

            value = request.form.get(
                column
            )

            input_data[column] = value

        input_df = pd.DataFrame(
            [input_data]
        )

        # Convert numeric-looking values
        for column in input_df.columns:

            try:

                input_df[column] = pd.to_numeric(
                    input_df[column]
                )

            except Exception:

                pass

        # One-hot encoding
        input_encoded = pd.get_dummies(
            input_df,
            drop_first=False
        )

        # Match training columns
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

            if label_encoder is not None:

                try:

                    prediction_value = label_encoder.inverse_transform(
                        prediction.astype(int)
                    )[0]

                except Exception:

                    prediction_value = prediction[0]

            else:

                prediction_value = prediction[0]

        else:

            prediction_value = prediction[0]

        return render_template(
            "result.html",
            prediction=prediction_value,
            task=task,
            target_column=target_column
        )

    except Exception as e:

        flash(
            f"Prediction failed: {e}"
        )

        return redirect(
            url_for("index")
        )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "application": "ML Data Analysis Dashboard"
    }


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

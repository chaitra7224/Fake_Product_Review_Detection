from flask import Flask, render_template, request, redirect, url_for, session
import os
import json
import joblib
import pandas as pd


app = Flask(__name__)

# Flask session secret key
app.secret_key = "fake_review_detection_secret_key"

# PROJECT PATHS

MODEL_FOLDER = "models"
DATASET_PATH = os.path.join("dataset", "review_dataset.csv")

# LOAD TRAINED MACHINE LEARNING FILES

vectorizer = joblib.load(
    os.path.join(MODEL_FOLDER, "tfidf_vectorizer.pkl")
)

decision_tree = joblib.load(
    os.path.join(MODEL_FOLDER, "decision_tree_model.pkl")
)

adaboost = joblib.load(
    os.path.join(MODEL_FOLDER, "adaboost_model.pkl")
)


# ---------------------------------------------------------
# LOAD PERFORMANCE RESULTS
# ---------------------------------------------------------

with open(
    os.path.join(MODEL_FOLDER, "performance.json"),
    "r"
) as file:

    performance = json.load(file)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ---------------------------------------------------------
# LOGIN
# ---------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        print("USERNAME:", username)
        print("PASSWORD:", password)

        if username and password:

            session["logged_in"] = True
            session["username"] = username

            print("SESSION:", session)

            return redirect("/upload")

        return render_template(
            "login.html",
            error="Please enter username and password."
        )

    return render_template("login.html")


# ---------------------------------------------------------
# DATASET UPLOAD AND PREVIEW
# ---------------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():

    # Login required
    if not session.get("logged_in"):

        return redirect(url_for("login"))


    if request.method == "POST":

        try:

            uploaded_file = request.files.get("dataset")


            # If user selected a CSV file
            if uploaded_file and uploaded_file.filename:

                if not uploaded_file.filename.lower().endswith(".csv"):

                    return render_template(
                        "upload.html",
                        error="Please select a CSV file."
                    )


                df = pd.read_csv(uploaded_file)

                file_name = uploaded_file.filename


            # If no file is selected, use project dataset
            else:

                if not os.path.exists(DATASET_PATH):

                    return render_template(
                        "upload.html",
                        error="Dataset file not found."
                    )


                df = pd.read_csv(DATASET_PATH)

                file_name = "reviews.csv"


            # Replace missing values
            df = df.fillna("")


            # First 50 rows
            preview = df.head(50).to_dict(
                orient="records"
            )


            return render_template(
                "upload.html",
                preview=preview,
                rows=len(df),
                columns=len(df.columns),
                column_names=list(df.columns),
                file_name=file_name
            )


        except Exception as e:

            return render_template(
                "upload.html",
                error=f"Unable to read CSV file: {e}"
            )


    return render_template("upload.html")


# ---------------------------------------------------------
# SINGLE REVIEW PREDICTION
# ---------------------------------------------------------

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    # Login required
    if not session.get("logged_in"):

        return redirect(url_for("login"))


    result = None
    confidence = None
    review = ""
    model_display = None
    selected_model = "adaboost"


    if request.method == "POST":

        review = request.form.get(
            "review",
            ""
        ).strip()


        selected_model = request.form.get(
            "model",
            "adaboost"
        )


        if review:

            # Convert review into TF-IDF features
            review_vector = vectorizer.transform(
                [review]
            )


            # Select model
            if selected_model == "decision_tree":

                model = decision_tree

                model_display = "Decision Tree"

            else:

                model = adaboost

                model_display = "AdaBoost"


            # Make prediction
            prediction_value = model.predict(
                review_vector
            )[0]


            # Calculate confidence
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    review_vector
                )[0]

                confidence = round(
                    max(probabilities) * 100,
                    2
                )


            # Convert prediction to readable result
            if prediction_value == 1:

                result = "Fake / Computer Generated Review"

            else:

                result = "Original / Genuine Review"


    return render_template(
        "prediction.html",
        result=result,
        confidence=confidence,
        review=review,
        model=model_display,
        selected_model=selected_model
    )


# ---------------------------------------------------------
# PERFORMANCE
# ---------------------------------------------------------

@app.route("/performance")
def performance_page():

    # Login required
    if not session.get("logged_in"):

        return redirect(url_for("login"))


    return render_template(
        "performance.html",
        performance=performance
    )


# ---------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------

@app.route("/analytics")
def analytics():

    # Login required
    if not session.get("logged_in"):

        return redirect(url_for("login"))


    if not os.path.exists(DATASET_PATH):

        return render_template(
            "analytics.html",
            error="Dataset file not found.",
            performance=performance
        )


    df = pd.read_csv(DATASET_PATH)


    # Clean labels
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    # Dataset class distribution
    label_counts = (
        df["label"]
        .value_counts()
        .to_dict()
    )


    # Convert rating to numeric
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )


    df = df.dropna(
        subset=["rating"]
    )


    df["rating"] = df["rating"].astype(int)


    # Rating versus authenticity
    rating_counts = (
        df.groupby(
            ["rating", "label"]
        )
        .size()
        .unstack(fill_value=0)
    )


    rating_data = []


    for rating in rating_counts.index:

        rating_data.append(
            {
                "rating": int(rating),

                "CG": int(
                    rating_counts.loc[rating].get(
                        "CG",
                        0
                    )
                ),

                "OR": int(
                    rating_counts.loc[rating].get(
                        "OR",
                        0
                    )
                )
            }
        )


    return render_template(
        "analytics.html",
        label_counts=label_counts,
        rating_data=rating_data,
        performance=performance
    )


# ---------------------------------------------------------
# LOGOUT
# ---------------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)
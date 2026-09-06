import os
import joblib
import pandas as pd

def test_model_file_exists():
    assert os.path.exists("model.pkl")

def test_model_loads():
    model = joblib.load("model.pkl")
    assert model is not None
def test_prediction_output():
    model = joblib.load("model.pkl")

    sample = pd.DataFrame([{
        "CGPA": 8.5,
        "Attendance": 90,
        "CodingScore": 85,
        "Projects": 3,
        "Internship": 1
    }])

    pred = model.predict(sample)[0]

    assert pred == 5      # Wrong intentionally

def test_dataset_exists():
    assert os.path.exists("data/student_placement.csv")
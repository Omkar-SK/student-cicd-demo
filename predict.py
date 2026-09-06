import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

print("Enter Student Details")

cgpa = float(input("CGPA: "))
attendance = int(input("Attendance: "))
coding = int(input("Coding Score: "))
projects = int(input("Projects: "))
internship = int(input("Internship (0/1): "))

sample = pd.DataFrame([{
    "CGPA": cgpa,
    "Attendance": attendance,
    "CodingScore": coding,
    "Projects": projects,
    "Internship": internship
}])

prediction = model.predict(sample)[0]

if prediction == 1:
    print("Prediction: PLACED")
else:
    print("Prediction: NOT PLACED")
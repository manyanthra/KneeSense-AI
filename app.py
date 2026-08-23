from flask import Flask, render_template, request
import os
import cv2
import pandas as pd

from predict import load_bundle, predict, KL_NAMES

app = Flask(__name__)

UPLOAD = "static/uploads"
RESULT = "static/results"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULT, exist_ok=True)

bundle = load_bundle("kl_classifier_final.joblib")

db = pd.read_csv("implant_db.csv")


def measure_anatomy(image):

    img = cv2.imread(image, 0)
    h, w = img.shape

    femur_y = int(h * 0.38)
    tibia_y = int(h * 0.60)

    femur = round((w * 0.50) * 0.12, 2)
    tibia = round((w * 0.46) * 0.12, 2)
    meniscus = round((tibia_y - femur_y) * 0.12, 2)

    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.line(color, (60, femur_y), (w-60, femur_y), (255,0,0), 2)
    cv2.line(color, (60, tibia_y), (w-60, tibia_y), (0,255,0), 2)
    cv2.arrowedLine(color, (w//2, femur_y), (w//2, tibia_y), (0,0,255), 2)

    out = os.path.join(RESULT, "analysis.png")
    cv2.imwrite(out, color)

    db["score"] = abs(db["FemurWidth"]-femur) + abs(db["TibiaWidth"]-tibia)
    implant = int(db.sort_values("score").iloc[0]["Size"])

    return meniscus, femur, tibia, implant, out


@app.route("/", methods=["GET","POST"])
def home():

    report = None

    if request.method == "POST":

        file = request.files["image"]

        if file.filename != "":

            path = os.path.join(UPLOAD, file.filename)
            file.save(path)

            grade, proba = predict(path, bundle)

            meniscus, femur, tibia, implant, img = measure_anatomy(path)

            report = {
                "patient": request.form["patient"],
                "age": request.form["age"],
                "sex": request.form["sex"],
                "grade": KL_NAMES[grade],
                "confidence": round(max(proba)*100,2),
                "meniscus": meniscus,
                "femur": femur,
                "tibia": tibia,
                "implant": implant,
                "image": img
            }

    return render_template("index.html", report=report)


if __name__ == "__main__":
    app.run(debug=True)
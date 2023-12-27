from flask import Flask, render_template, request
import joblib
from Spam_classifier2 import prepare_pipeline

app = Flask(__name__)
model = joblib.load("RandomForest_best.joblib")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    email = request.form['email']
    processed_email = prepare_pipeline.transform([email])
    probability = model.predict_proba(processed_email)[0][1] * 100  # Convert to percentage
    probability = "{:.2f}".format(probability)
    return render_template('index.html', probability=probability)

if __name__ == "__main__":
    app.run(debug=True)


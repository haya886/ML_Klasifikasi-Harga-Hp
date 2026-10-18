# TAHAP MONITORING DAN DEPLOY
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model yang telah disimpan
joblib_model = joblib.load('gb_model.joblib') # Pastikan path file sesuai dengan penyimpanan Anda

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = joblib_model.predict(data)
    return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
    app.run(debug=True)
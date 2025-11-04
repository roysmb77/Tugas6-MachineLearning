from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # penting untuk menyimpan session

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'random_forest_placement.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    prediction = session.pop('prediction', None)
    probability = session.pop('probability', None)
    error = session.pop('error', None)
    return render_template('index.html', prediction=prediction, probability=probability, error=error)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        iq = float(request.form['iq'])
        prev_sem = float(request.form['prev_sem'])
        cgpa = float(request.form['cgpa'])
        academic_perf = float(request.form['academic_perf'])
        internship = 1 if request.form['internship'] == 'Yes' else 0
        extracurricular = float(request.form['extracurricular'])
        communication = float(request.form['communication'])
        projects = int(request.form['projects'])
        trend = prev_sem - cgpa

        features = np.array([[iq, prev_sem, cgpa, academic_perf, internship,
                              extracurricular, communication, projects, trend]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        result_text = "🎓 Selamat! Mahasiswa berpotensi besar ditempatkan." if prediction == 1 else "⚠️ Mahasiswa belum memenuhi kriteria penempatan."

        # Simpan hasil ke session, lalu redirect ke halaman utama
        session['prediction'] = result_text
        session['probability'] = f"{probability:.2f}%"
        return redirect(url_for('index'))

    except Exception as e:
        session['error'] = f"Terjadi kesalahan: {str(e)}"
        return redirect(url_for('index'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080)) app.run(host="0.0.0.0", port=port)


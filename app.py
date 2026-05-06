from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import pickle
import numpy as np
import json
from werkzeug.utils import secure_filename
from feat_extract import extract_features, is_valid_fingerprint

app = Flask(__name__)
app.secret_key = 'diabtrace_suiit_burla_2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# ─── In-memory user store (simple, no DB needed) ───────────────────────────
users = {}          # username -> password
predictions = {}    # username -> list of prediction dicts

# ─── Load ML Model ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

FEATURE_NAMES = ['Fingerprint_Type_enc', 'Ridge_Count', 'Ridge_Density', 'Age', 'BMI', 'Family_History_enc']
FEATURE_IMPORTANCES = {
    'BMI': 0.1561,
    'Ridge Count': 0.1103,
    'Age': 0.1053,
    'Ridge Density': 0.2110,
    'Fingerprint Type': 0.3749,
    'Family History': 0.0423
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_risk(fp_type, ridge_count, ridge_density, age, bmi, family_history):
    """Run prediction using trained Random Forest model"""
    type_map = {'arch': 0, 'loop': 1, 'whorl': 2}
    fh_map = {'no': 0, 'yes': 1}

    fp_enc = type_map.get(fp_type.lower(), 1)
    fh_enc = fh_map.get(family_history.lower(), 0)

    X = np.array([[fp_enc, int(ridge_count), float(ridge_density),
                   int(age), float(bmi), fh_enc]])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}

    return pred, proba_dict

# ─── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', user=session.get('user'))

@app.route('/about')
def about():
    return render_template('about.html', user=session.get('user'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash('Username and password are required.', 'error')
        elif username in users:
            flash('Username already exists. Please choose another.', 'error')
        else:
            users[username] = password
            predictions[username] = []
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html', user=session.get('user'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username in users and users[username] == password:
            session['user'] = username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('upload_fingerprint'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html', user=session.get('user'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/upload_fingerprint', methods=['GET', 'POST'])
def upload_fingerprint():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'fingerprint' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)

        file = request.files['fingerprint']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Upload .bmp, .png, .jpg etc.', 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Validate that the image is actually a fingerprint
        valid, reason = is_valid_fingerprint(filepath)
        if not valid:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'Invalid image: {reason} Please upload a real fingerprint scan (.bmp, .png, .jpg).'
            })

        # Extract features using OpenCV
        features = extract_features(filepath)

        # Store in session for Step 2
        session['fp_features'] = features
        session['fp_filename'] = filename

        return jsonify({
            'success': True,
            'pattern_type': features['pattern_type'].capitalize(),
            'ridge_count': features['ridge_count'],
            'ridge_density': features['ridge_density']
        })

    return render_template('upload_fingerprint.html', user=session.get('user'))

@app.route('/prediction_form')
def prediction_form():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))

    fp_features = session.get('fp_features')
    if not fp_features:
        flash('Please upload a fingerprint first.', 'error')
        return redirect(url_for('upload_fingerprint'))

    return render_template('prediction_form.html',
                           user=session.get('user'),
                           fp_features=fp_features)

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))

    fp_features = session.get('fp_features', {})
    fp_type = fp_features.get('pattern_type', 'loop')
    ridge_count = fp_features.get('ridge_count', 20)
    ridge_density = fp_features.get('ridge_density', 12.0)

    age = request.form.get('age', 30)
    bmi = request.form.get('bmi', 22.0)
    family_history = request.form.get('family_history', 'no')

    try:
        age = int(age)
        bmi = float(bmi)
    except (ValueError, TypeError):
        flash('Invalid age or BMI value.', 'error')
        return redirect(url_for('prediction_form'))

    risk, proba = predict_risk(fp_type, ridge_count, ridge_density, age, bmi, family_history)

    # Store in user history
    record = {
        'fp_type': fp_type.capitalize(),
        'ridge_count': ridge_count,
        'ridge_density': ridge_density,
        'age': age,
        'bmi': bmi,
        'family_history': family_history.capitalize(),
        'risk': risk,
        'proba': proba
    }
    username = session['user']
    if username not in predictions:
        predictions[username] = []
    predictions[username].append(record)

    # Clear session FP data
    session.pop('fp_features', None)

    return render_template('result.html',
                           user=username,
                           record=record,
                           proba=proba,
                           feature_importance=FEATURE_IMPORTANCES)

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))

    user_preds = predictions.get(session['user'], [])
    return render_template('dashboard.html',
                           user=session.get('user'),
                           predictions=user_preds)

@app.route('/remidy')
def remidy():
    if 'user' not in session:
        return redirect(url_for('login'))
    risk = request.args.get('diabetes_risk', 'low').upper()
    return render_template('suggestions.html', user=session.get('user'), risk=risk)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)

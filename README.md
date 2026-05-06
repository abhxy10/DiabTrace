# DiabTrace Analytics — Setup & Run Guide
# B.Tech Final Year Project | SUIIT Burla | CSE 2026
# Roll No: 23LBTICS01 | Abhay Singh

## SETUP (First Time Only)

# 1. Make sure Python 3.x is installed
python --version

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (uses the dataset in /dataset/ folder)
python train_model.py

# 4. Run the app
python app.py

# 5. Open browser at: http://127.0.0.1:5000


## PROJECT STRUCTURE

diabtrace/
├── app.py                  # Main Flask application
├── feat_extract.py         # OpenCV fingerprint feature extraction
├── train_model.py          # Model training script
├── requirements.txt
├── models/
│   └── model.pkl           # Trained Random Forest model
├── dataset/
│   └── fingerprint_diabetes_dataset_3000.xlsx
├── templates/              # HTML pages
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── login.html
│   ├── upload_fingerprint.html
│   ├── prediction_form.html
│   ├── result.html
│   ├── dashboard.html
│   ├── suggestions.html
│   └── about.html
└── uploads/                # Temporary fingerprint uploads


## HOW TO USE

1. Go to http://127.0.0.1:5000
2. Register a new account
3. Log in
4. Click "Predict Risk" in nav
5. Upload a fingerprint image (.bmp / .png / .jpg)
6. Click "Extract Features" — OpenCV will analyse the image
7. Click "Proceed to Prediction"
8. Enter Age, BMI, Family History
9. Click "Predict Diabetes Risk"
10. View result with risk level + probabilities + feature importance


## MODEL INFO

- Algorithm: Random Forest Classifier (200 trees)
- Dataset: 3000 records (synthetic, based on clinical literature)
- Features: Fingerprint Type, Ridge Count, Ridge Density, Age, BMI, Family History
- Test Accuracy: 97.8%
- Classes: Low / Medium / High risk

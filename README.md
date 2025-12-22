🎓 Student Performance Prediction – End-to-End ML Project

An end-to-end Machine Learning project that predicts a student’s Math score based on demographic and academic features.
The project follows industry-standard ML pipeline practices and includes training, prediction, and a Flask web application.

📌 Project Overview

This project demonstrates how a real-world ML system is built — not just a notebook model.

✔ What this project does

Ingests raw student data

Applies data preprocessing & feature engineering

Trains and evaluates multiple ML models

Selects and saves the best model

Serves predictions through a Flask web app

🧠 Problem Statement

Predict the Math score of a student using:

Gender

Race/Ethnicity

Parental level of education

Lunch type

Test preparation course

Reading score

Writing score

🏗️ Project Architecture
Project for ML/
│
├── artifacts/                  # Saved models & preprocessors
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebook/
│   └── data/
│       └── stud.csv             # Raw dataset
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── app.py                      # Flask app
├── requirements.txt
└── README.md

🔄 ML Pipeline Stages
1️⃣ Data Ingestion

Reads raw CSV data

Stores a raw copy

Splits data into train and test sets

Saves them in the artifacts/ directory

File: data_ingestion.py

2️⃣ Data Transformation

Handles missing values

Encodes categorical variables (OneHotEncoder)

Scales numerical features

Saves preprocessing pipeline as preprocessor.pkl

File: data_transformation.py

3️⃣ Model Trainer

Trains multiple regression models

Performs hyperparameter tuning (GridSearchCV)

Evaluates models using R² score

Saves the best model as model.pkl

File: model_trainer.py

4️⃣ Prediction Pipeline

Loads saved model & preprocessor

Transforms user input

Returns prediction

File: predict_pipeline.py

5️⃣ Web Application

Built using Flask

Accepts user input via HTML forms

Displays predicted Math score

File: app.py

🚀 Models Used

Linear Regression

K-Nearest Neighbors Regressor

Decision Tree Regressor

Random Forest Regressor

AdaBoost Regressor

Gradient Boosting Regressor

The best model is selected automatically based on R² score.

🧪 Evaluation Metric

R² Score (Coefficient of Determination)

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone <your-repo-url>
cd Project-for-ML

2️⃣ Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run training pipeline
python -m src.components.data_ingestion


This will generate:

artifacts/
├── model.pkl
└── preprocessor.pkl

5️⃣ Run Flask application
python app.py


Open browser:

http://127.0.0.1:5000

🧠 Key Design Highlights

Modular ML pipeline

Custom exception handling

Centralized logging

Reusable utility functions

No data leakage

Production-ready structure

📦 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Flask

HTML/CSS

Pickle / Dill

📈 Future Improvements

Add Docker support

Deploy on AWS / Render / Heroku

Add FastAPI version

Improve UI

Add CI/CD pipeline

👤 Author

Omkar Kadam
Computer Engineering Student
Interested in Machine Learning & MLOps

⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork!
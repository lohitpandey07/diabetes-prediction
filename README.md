# 🩺 Diabetes Prediction Web App

A machine learning web app built with **Streamlit** that predicts whether a person is
diabetic or not from 8 medical measurements. The model is a Support Vector Machine (SVM)
trained in a Jupyter notebook and saved as a `.sav` file.

> ⚠️ **Disclaimer:** For educational purposes only. This is not a medical diagnosis.
> Always consult a qualified doctor for health decisions.

**Live demo:** diabetes-prediction-gmxmvccxgntybkpepo45iu.streamlit.app

## Features

- Simple form to enter patient details
- Instant prediction: diabetic or not diabetic
- Input validation, so invalid or empty values show a friendly error
- Model trained and evaluated in a documented notebook (`train.ipynb`)

## Project structure

```
diabetes-prediction/
├── Diabetes_Prediction_Web_app.py     # Streamlit web app
├── train.ipynb                        # Notebook: data exploration, training, evaluation, saving the model
├── requirements.txt                   # Python dependencies
├── README.md
├── data/
│   └── diabetes_dataset_800_records.csv   # Dataset (800 records)
└── model/
    └── trained_model.sav              # Saved SVM model used by the app
```

## Dataset

`data/diabetes_dataset_800_records.csv` contains **800 records** and 9 columns.

| Column                   | Description                                |
| ------------------------ | ------------------------------------------ |
| Pregnancies              | Number of pregnancies                      |
| Glucose                  | Glucose level                              |
| BloodPressure            | Blood pressure                             |
| SkinThickness            | Skin thickness                             |
| Insulin                  | Insulin level                              |
| BMI                      | Body mass index                            |
| DiabetesPedigreeFunction | Family history score                       |
| Age                      | Age in years                               |
| **Diabetic**             | **Target:** 0 = not diabetic, 1 = diabetic |

Class distribution: 704 diabetic (1) and 96 not diabetic (0).

## Model

| Item              | Value                                         |
| ----------------- | --------------------------------------------- |
| Algorithm         | Support Vector Machine (`SVC`, linear kernel) |
| Train/test split  | 70% / 30% (stratified, `random_state=42`)     |
| Training accuracy | 92.3%                                         |
| Test accuracy     | 92.9%                                         |

### Limitations

- The dataset is **imbalanced**: about 88% of the records are diabetic. A model that always
  predicted "diabetic" would already score 88% accuracy, so the 92.9% test accuracy is only
  a modest improvement over that baseline.
- Accuracy alone does not show how well the "not diabetic" class is detected. Checking
  precision, recall and a confusion matrix would give a fuller picture.
- This is a small dataset and a simple model, so predictions should not be relied on for real decisions.

## Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the app
streamlit run Diabetes_Prediction_Web_app.py
```

### Retrain the model (optional)

Open `train.ipynb` in Jupyter or VS Code and run all cells. It reads the dataset from
`data/` and saves the new model to `model/trained_model.sav`.

## How it works

1. `train.ipynb` loads the dataset, splits it into training and test sets, trains an SVM, and
   saves it with `pickle` to `model/trained_model.sav`.
2. The Streamlit app loads the saved model.
3. The user enters 8 values, which are converted to numbers and placed in a DataFrame.
4. The model returns 0 (not diabetic) or 1 (diabetic), which is shown on screen.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (keep `model/trained_model.sav` included).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Select your repository and set the main file to `Diabetes_Prediction_Web_app.py`.
4. Click **Deploy**.

> The model was trained with scikit-learn 1.7.2, which is pinned in `requirements.txt`
> so the saved model loads correctly.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, Jupyter Notebook

# 🩺 MediCare AI — Personalized Healthcare & Medicine Recommendation System

## Overview

**MediAI** is an educational and informational machine learning application that combines disease prediction, disease-wise medication information, patient-profile analysis support, and content-based medicine similarity into a single Streamlit dashboard.

The system allows a user to:

- Select symptoms from the actual 132 symptom features used by the trained disease model.
- Predict a disease using a saved Linear SVC model.
- View disease-wise medication information from the provided medication dataset.
- Explore similar medicine profiles using TF-IDF and cosine similarity.
- Understand the datasets, machine learning pipeline, and project limitations through an interactive dashboard.

> ⚠️ **Important:** This project is an educational/informational prototype. It does not provide medical diagnosis, prescribe medication, or replace advice from a qualified healthcare professional.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Build a machine learning model for disease prediction from symptom features.
2. Connect predicted diseases with disease-wise medication information.
3. Develop a content-based medicine similarity system.
4. Build an interactive Streamlit dashboard around the developed ML components.
5. Demonstrate the complete workflow from data preprocessing to deployment.
6. Clearly document limitations and data-quality issues in the provided datasets.

---

## 🧠 System Architecture

### Disease Prediction Pipeline

```text
Patient Symptoms
       ↓
132 Binary Symptom Features
       ↓
Saved Linear SVC Model
       ↓
Predicted Disease
       ↓
medications.csv
       ↓
Disease-wise Medication Information
```

### Medicine Similarity Pipeline

```text
Medicine_Details.csv
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Cosine Similarity
       ↓
Similar Medicine Profiles
```

---

## 📊 Dataset Information

The project uses four different datasets. Each dataset has a separate role.

| Dataset | Purpose | Records |
|---|---|---:|
| `Training.csv` | Disease prediction from symptoms | 4,920 |
| `medications.csv` | Disease-wise medication information | 41 |
| `Cleaned_Dataset.csv` | Patient-profile/risk-analysis support | 349 |
| `Medicine_Details.csv` | Medicine-level text similarity | 11,825 |

### 1. Training.csv

This is the main disease prediction dataset.

- 4,920 records
- 132 symptom features
- 1 target column: `prognosis`
- 41 disease classes
- Symptom features are binary (`0` / `1`)

The saved disease prediction model uses the same 132 symptom features.

### 2. medications.csv

This dataset contains disease-wise medication information.

It is used after disease prediction to retrieve medication information associated with the predicted disease.

The dataset does **not** contain patient-level medication outcomes or dosage information.

Therefore, the application does not learn or prescribe an optimal medication for an individual patient.

### 3. Cleaned_Dataset.csv

This dataset contains patient-profile and risk-related information such as:

- Age
- Gender
- Blood pressure
- Cholesterol level
- Disease
- Risk level
- Outcome variable

It is used as supporting patient-profile/risk-analysis data.

It is **not artificially merged with `Training.csv`**, because the records do not represent the same patient-level observations.

### 4. Medicine_Details.csv

This dataset contains medicine-level information including:

- Medicine Name
- Composition
- Uses
- Side effects
- Manufacturer
- Review percentages
- Image URL

For medicine similarity, the system uses:

- Medicine Name
- Composition
- Uses
- Side effects

Review percentages are not used to determine medical effectiveness or to rank medicines.

---

## 🤖 Machine Learning

### Disease Prediction

The disease prediction component uses a saved **Linear SVC (Support Vector Classifier)** model.

The model is loaded from:

```text
models/best_model.pkl
```

The disease labels are decoded using:

```text
models/disease_encoder.pkl
```

The model receives:

```text
132 binary symptom features
```

and predicts one of:

```text
41 disease classes
```

### Model Evaluation

On the provided dataset and evaluation setup, the model achieved:

- Test accuracy: **100%**
- Precision/Recall/F1: **1.00** across the evaluated classes

However, these results should **not** be interpreted as real-world diagnostic accuracy.

The provided `Training.csv` contains 4,920 rows but only 304 unique rows after removing exact duplicates. Therefore, identical records can appear in both training and test splits, which can inflate the measured performance.

For this reason, the reported performance should be understood as performance on the provided dataset and split rather than clinical diagnostic performance.

---

## 💊 Medicine Recommendation / Information

After the disease prediction step:

```text
Predicted Disease
       ↓
medications.csv
       ↓
Medication Information
```

The application displays the medication information available for the predicted disease.

The system does not:

- prescribe medication
- provide dosage instructions
- determine an optimal medicine
- use patient-level treatment outcomes
- replace professional medical advice

---

## 🔎 Medicine Similarity

The medicine similarity component uses a **content-based filtering approach**.

The text fields are combined from:

```text
Medicine Name
Composition
Uses
Side effects
```

The text is converted into numerical vectors using:

```text
TF-IDF
```

Similarity between medicine profiles is then calculated using:

```text
Cosine Similarity
```

### Example workflow

```text
Selected Medicine
       ↓
Medicine Text Profile
       ↓
TF-IDF Vector
       ↓
Cosine Similarity
       ↓
Top Similar Medicine Profiles
```

A higher similarity score indicates greater **textual similarity** between medicine records.

It does **not** mean:

- clinical equivalence
- equal effectiveness
- equal safety
- interchangeability
- suitability for a particular patient

---

## 🖥️ Streamlit Dashboard

The final application is built using **Streamlit**.

### Dashboard Pages

#### 🏠 Diagnosis

The Diagnosis page provides:

- Patient profile fields
- Searchable symptom selection
- 132 actual symptom features
- Selected symptom counter
- Disease prediction
- Model confidence
- Top model probabilities
- Patient profile summary
- Disease-wise medication information
- Medical disclaimer

#### 💊 Medicine Similarity

The Medicine Similarity page allows users to:

- Select a medicine
- Choose the number of similar medicines
- Calculate textual similarity
- View similarity scores
- View composition
- View uses
- View side effects

#### 📊 Project Overview

The Project Overview page explains:

- System architecture
- Dataset roles
- Number of disease classes
- Number of symptom features
- Medicine records
- Machine learning components
- Data-quality observations
- Project limitations

#### ℹ️ About

The About page explains:

- Project objective
- Disease prediction
- Medication information
- Medicine similarity
- Technologies used
- Educational purpose and limitations

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Support Vector Classifier (Linear SVC)
- LabelEncoder

### Medicine Similarity

- TF-IDF Vectorizer
- Cosine Similarity

### Model Persistence

- Joblib

### Dashboard

- Streamlit
- Custom CSS

### Development

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## 📁 Project Structure

```text
personalized-healthcare-recommendation-system/
│
├── data/
│   ├── Training.csv
│   ├── Cleaned_Dataset.csv
│   ├── medications.csv
│   └── Medicine_Details.csv
│
├── models/
│   ├── best_model.pkl
│   └── disease_encoder.pkl
│
├── notebook/
│   ├── 01_Data_Inspection.ipynb
│   ├── 02_Disease_Prediction.ipynb
│   ├── 03_Medicine_Recommendation.ipynb
│   ├── 04_Final_Evaluation.ipynb
│   └── 05_Medicine_Similarity.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 End-to-End Workflow

```text
                    ┌─────────────────────┐
                    │   Patient Symptoms  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ 132 Binary Features│
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Linear SVC Model   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Predicted Disease   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  medications.csv    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Medication Info     │
                    └─────────────────────┘


              Medicine Similarity Module

                    Medicine Details
                           ↓
                    Text Preprocessing
                           ↓
                         TF-IDF
                           ↓
                  Cosine Similarity
                           ↓
              Similar Medicine Profiles
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Open the project

```bash
cd personalized-healthcare-recommendation-system
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

#### Windows

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 📓 Project Development Stages

The project was developed through separate stages:

### 01 — Data Inspection

Dataset structure, data types, missing values, duplicates, symptom features and disease distribution were examined.

### 02 — Disease Prediction

The disease prediction model was trained and evaluated using the symptom features from `Training.csv`.

### 03 — Medicine Recommendation

Disease prediction was connected to the disease-wise medication dataset.

### 04 — Final Evaluation

The complete disease prediction and medication-information pipeline was evaluated.

### 05 — Medicine Similarity

A content-based medicine similarity system was developed using TF-IDF and cosine similarity.

These notebooks are development and analysis artifacts; the final user-facing interface is the Streamlit application in `app.py`.

---

## ⚠️ Limitations

This project has several important limitations.

### Dataset limitations

- `Training.csv` contains many duplicate records.
- The reported 100% disease-model accuracy may be inflated because duplicate records can appear across training and test splits.
- The model performance applies only to the provided dataset and evaluation setup.
- It should not be interpreted as real-world diagnostic accuracy.

### Medication limitations

- `medications.csv` provides disease-wise medication information.
- It does not contain patient-level medication outcomes.
- It does not support learning the optimal medication for an individual.
- The application does not provide dosage instructions.

### Medicine similarity limitations

- Similarity is based on textual information.
- Similarity does not establish clinical equivalence.
- Similarity does not establish safety or effectiveness.
- Similar medicines should not be considered automatically interchangeable.

### Dataset separation

`Cleaned_Dataset.csv` and `Training.csv` represent different datasets with different structures and purposes.

Therefore, the project does not artificially merge their patient records.

---

## 🔐 Educational Disclaimer

> ⚠️ **This system is intended for educational and informational purposes only.**
>
> It does not provide medical diagnosis, prescribe medication, or replace advice from a qualified healthcare professional.
>
> Disease prediction results are based on the provided machine learning dataset.
>
> Medication information is retrieved from the provided disease-wise dataset.
>
> Medicine similarity represents textual similarity and does not imply clinical equivalence or interchangeability.

---

## 👨‍💻 Project

**MediAI — Personalized Healthcare & Medicine Recommendation System**

Built using:

```text
Python • Pandas • NumPy • Scikit-learn • Joblib
TF-IDF • Cosine Similarity • Streamlit
```

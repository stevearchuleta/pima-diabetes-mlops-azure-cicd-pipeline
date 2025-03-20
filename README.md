```md
# PIMA DIABETES PREDICTION - AZURE MLOPS PIPELINE

## 1. Project Overview
This project provides an **MLOps pipeline for predicting the likelihood of diabetes** in patients using a machine learning model. The model is trained using an **XGBoost Classifier**, which analyzes medical attributes such as **glucose levels, blood pressure, BMI, and age**.

The pipeline is fully automated using **Azure Machine Learning (Azure ML)** for model training, evaluation, and deployment. **GitHub Actions is used for CI/CD automation**, ensuring that each model update is seamlessly integrated into the pipeline.

### Dataset Description
- **File Name:** `pima.csv`
- **Columns:**  
  - `Outcome`: Target variable (binary, 1 = Diabetes, 0 = No Diabetes).  
  - `Glucose`: Plasma glucose concentration.  
  - `BloodPressure`: Diastolic blood pressure (mm Hg).  
  - `SkinThickness`: Triceps skin fold thickness (mm).  
  - `Insulin`: 2-hour serum insulin (mu U/ml).  
  - `BMI`: Body Mass Index.  
  - `DiabetesPedigreeFunction`: A function that scores diabetes likelihood based on genetic factors.  
  - `Age`: Age of the individual in years.  
- **Data Size:** Approximately **8,000 patient records** collected from clinical studies.

---

## 2. Installation and Setup

### 1️⃣ Install Dependencies
Run the `setup.sh` script to install required Python dependencies:
```sh
bash setup.sh
```

### 2️⃣ Ensure Azure ML CLI is Installed
```sh
az extension add -n ml -y
```

---

## 3. Pipeline Workflow
The MLOps pipeline consists of the following key stages:

### 1️⃣ Preprocessing (`prep.py`)
- Loads `pima.csv` and cleans the data.
- Handles **missing values** using median imputation.
- Scales numerical features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `Age`).

### 2️⃣ Training (`train.py`)
- Trains an **XGBoost Classifier** model on the preprocessed dataset.
- Uses **Accuracy, Precision, Recall, F1-score, and AUC-ROC** as evaluation metrics.

### 3️⃣ Evaluation (`train.py`)
- The model’s performance is **logged in MLflow**.
- Evaluation metrics can be accessed in **Azure ML Studio**.

### 4️⃣ Model Registration (`register.py`)
- Saves the trained model to **Azure ML** for deployment.
- Stores metadata about the registered model.

**Azure ML orchestrates job execution**, ensuring each step runs in the correct sequence.

---

## 4. How to Run the Pipeline

### Submit the MLOps Pipeline to Azure ML
Use the following command to submit the pipeline job:
```sh
bash run-job.sh mlops/azureml/train/newpipeline.yml pima-diabetes-experiment
```
This command **submits an Azure ML job**, triggering preprocessing, training, evaluation, and model registration.

---

## 5. Folder and File Structure
This project follows **Azure MLOps best practices** with the following directory structure:

- `.github/workflows/` → Contains **CI/CD pipeline definitions** (`ci.yml`, `deploy-model-training-pipeline-classical.yml`).
- `data/` → Stores raw dataset (`pima.csv`).
- `data-science/src/` → Contains machine learning scripts (`prep.py`, `train.py`, `register.py`).
- `mlops/azureml/train/` → Stores Azure ML pipeline definitions (`newpipeline.yml`, `train.yml`).
- `jobs/pipeline/` → Contains `pipeline.yml`, which outlines the entire MLOps workflow.
- `requirements.txt` → Lists Python dependencies.
- `setup.sh` → Installs **Azure ML CLI tools**.
- `run-job.sh` → Submits the MLOps pipeline job.

---

## 6. CI/CD Automation with GitHub Actions
This project uses **GitHub Actions** to automate **model training and deployment**.

- `ci.yml` → Runs **pre-commit checks, unit tests, and formatting validation**.
- `deploy-model-training-pipeline-classical.yml` → Deploys the model to **Azure ML Pipelines**.
- Every push to `main` **triggers a training pipeline**.

---

## 7. Additional Notes
- This project follows **MLOps best practices for Azure**.
- Future improvements may include **hyperparameter tuning and real-time model monitoring**.
- To monitor jobs, navigate to **Azure ML Studio** and check job logs.
```
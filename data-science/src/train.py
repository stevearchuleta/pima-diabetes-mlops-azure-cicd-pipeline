# ==================================================
# TRAINING SCRIPT FOR PIMA DIABETES PREDICTION MODEL
# ==================================================
"""
Trains an XGBoost Classifier model for Pima Diabetes prediction and evaluates it using Accuracy, Precision, Recall, F1-score, and AUC-ROC.
"""

import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==============================
# PARSE ARGUMENTS FOR FILE PATHS
# ==============================
def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--max_depth', type=int, default=5, help="Maximum depth of the trees")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Boosting learning rate")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of boosting rounds")
    return parser.parse_args()

# ======================================
# MAIN FUNCTION - TRAIN & EVALUATE MODEL
# ======================================
def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # ==== READ TRAIN AND TEST DATA ====
    train_df = pd.read_csv(os.path.join(args.train_data, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.test_data, "test.csv"))

    # ==== SPLIT FEATURES AND TARGET ====
    y_train = train_df["Outcome"]
    X_train = train_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]
    X_test = test_df.drop(columns=["Outcome"])

    # ==== TRAIN XGBOOST CLASSIFIER ====
    model = xgb.XGBClassifier(max_depth=args.max_depth, learning_rate=args.learning_rate, n_estimators=args.n_estimators)
    model.fit(X_train, y_train)

    # ==== LOG MODEL PARAMETERS ====
    mlflow.log_param("model", "XGBoost Classifier")
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("n_estimators", args.n_estimators)

    # ==== PREDICT ON TEST DATA ====
    yhat_test = model.predict(X_test)

    # ==== COMPUTE EVALUATION METRICS ====
    accuracy = accuracy_score(y_test, yhat_test)
    precision = precision_score(y_test, yhat_test)
    recall = recall_score(y_test, yhat_test)
    f1 = f1_score(y_test, yhat_test)
    auc_roc = roc_auc_score(y_test, yhat_test)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

    # ==== LOG METRICS ====
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1-score", f1)
    mlflow.log_metric("AUC-ROC", auc_roc)

    # ==== SAVE THE MODEL ====
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

# ==============
# EXECUTE SCRIPT
# ==============
if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()

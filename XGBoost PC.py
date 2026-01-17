import os
import numpy as np
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier


# Load Dataset
path = kagglehub.dataset_download("johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer")

print("Path to dataset files:", path)
print("Files:", os.listdir(path))

csv_path=os.path.join(path,'Debernardi et al 2020 data.csv')
df=pd.read_csv(csv_path)

# Drop Columns
df=df.drop(columns=['sample_id','patient_cohort','sample_origin','stage','benign_sample_diagnosis'])

# Convert sex into numerical data
df["sex"] = df["sex"].map({"M":1, "F":0})

# Define Target

target_column='diagnosis'
X=df.drop(target_column,axis=1)
y=df[target_column]
y=y-1
class_names=["Healthy", "Benign", "Cancer"]

#Split Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build Model

model=XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    min_child_weight=1,
    gamma=0.0,
    random_state=42,
    n_jobs=-1
)

# Increase Weight of Cancer Mistakes

cancer_class=2
cancer_weight=50
sample_weight=np.where(y_train == cancer_class, cancer_weight, 1.0)

# Train Model

model.fit(X_train, y_train,sample_weight=sample_weight)

#Lower Threshold for Cancer Detection

cancer_threshold=0.25
y_proba = model.predict_proba(X_test)
y_pred_adjusted = np.argmax(y_proba,axis=1)
cancer_prob=y_proba[:,cancer_class]
y_pred_adjusted[cancer_prob >= cancer_threshold]=cancer_class

# Evaluate Model Accuracy
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_adjusted))
print("\nClassification report:\n", classification_report(
    y_test,
    y_pred_adjusted,
    target_names=class_names,
    digits=1
)
      )

#ROC-AUC Metric


roc_auc=roc_auc_score(y_test, y_proba,multi_class='ovr')
print(f"ROC-AUC (OvR): {roc_auc:.4f}")
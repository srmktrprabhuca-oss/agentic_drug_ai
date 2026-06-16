import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from utils.preprocessing import encode_drugs

df = pd.read_csv("data/drug_dataset.csv")

drug_encoded, drug_list = encode_drugs(df)

clinical_cols = ["glucose","bp","cholesterol","kidney","hba1c","bmi","hdl","ldl","ulcer"]

X = pd.concat([drug_encoded, df[clinical_cols]], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = XGBClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model/interaction_model.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")

print("Accuracy:", model.score(X_test, y_test))
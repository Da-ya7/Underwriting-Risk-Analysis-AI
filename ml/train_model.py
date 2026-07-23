import json
import joblib
import pandas as pd # to load csv
from sklearn.ensemble import RandomForestClassifier #imports the machine learning algorithm.
from sklearn.model_selection import train_test_split #Used to split the dataset. 1000 rows= 800(training),200(Testing)
from sklearn.metrics import classification_report,roc_auc_score #functions evaluate the trained model.

FEATURES = [
    "age", "annual_income", "sum_assured", "bmi", "smoker",
    "alcohol_consumption", "pre_existing_disease", "family_medical_history",
    "occupation_risk", "credit_score", "num_previous_claims", "years_with_insurer",
]
df=pd.read_csv("../data/underwriting_data.csv")
x=df[FEATURES] #This selects only the feature columns.[age,annual_income,...]
y=df["approved"]## Stores the correct answers (target variable) for training.

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,#20% of data goes into testing
    random_state=42,#it makes sure rows selected for testing would never change same random way
    stratify=y #computer keeps the same ratio for testing and training
)
"""stratify=y ensures the Approved/Rejected distribution stays the same in both the training set and the testing set."""

model=RandomForestClassifier(
    n_estimators=200,#Build 200 decision trees.
    random_state=42, #Every run builds the forest in the same random way just like before
)
model.fit(x_train,y_train) #Learn the relationship between the customer features (X_train) and the correct answers (y_train)

pred=model.predict(x_test)#X_test contains new customer details that the model has never seen during training
print(classification_report(y_test,pred))#
"""classification_report() counts all these matches and mistakes and calculates metrics like accuracy, precision, recall, and F1-score."""
print("ROC-AUC",roc_auc_score(y_test,model.predict_proba(x_test)[:,1])) #How well does the model separate approved customers from rejected customers based on their predicted probabilities
#[:,1]-Give me all rows, but only column 1.

joblib.dump(model,"model.pkl")

feature_meta={
    "features": FEATURES,
    "importances": dict(
    zip(
        FEATURES,
        model.feature_importances_.round(4).tolist()
    )#Rounds each number to 4 decimal places.,tolist()-Converts the NumPy array into a normal Python list.,zip() -pairs them together and dict()-convert them into dictionary
),#model.feature_importances_-  tells us how much each feature contributed to the Random Forest's decisions across the entire training dataset.
# Example:
# If Age has importance 0.35,
# it means Age contributed more to the model's decisions
# than features with lower importance.

    # direction: +1 means higher value = more risk, -1 means higher value = less risk
    # (not used by explain.py yet, kept here for reference/future use)
    "direction": {
        "age": 1, "annual_income": -1, "sum_assured": 0, "bmi": 1, "smoker": 1,
        "alcohol_consumption": 1, "pre_existing_disease": 1, "family_medical_history": 1,
        "occupation_risk": 1, "credit_score": -1, "num_previous_claims": 1,
        "years_with_insurer": -1,
    },

    # risk_thresholds: cutoff values explain.py uses to decide if a feature counts
    # as "risky" for a given applicant. REQUIRED - explain.py crashes without this key.
    "risk_thresholds": {
        "age": 55, "bmi": 30, "credit_score": 650, "num_previous_claims": 2,
        "years_with_insurer": 1,
    },
}
#feature_importances_ is learned from the training data, so if the data changes, the importances can change too.

with open("feature_meta.json", "w") as f:
    json.dump(feature_meta, f, indent=2)#This writes the dictionary to a file. and indent=2 argument. It simply makes the JSON nicely formatted and easier for humans to read.
print("Saved model.pkl + feature_meta.json")
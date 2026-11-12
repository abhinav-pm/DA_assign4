import sys
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC

# ---------------- Configuration ----------------
TRAIN_PATH = "Customer_train.csv"
TARGET = "Segmentation"

NUM_COLS = ["Age", "Work_Experience", "Family_Size"]
ORD_COLS = ["Spending_Score"]
ORD_ORDER = ["Low", "Average", "High"]
NOM_COLS = ["Gender", "Ever_Married", "Graduated", "Profession", "Var_1"]

# Binary SVM hyperparameters (placeholder values; to be updated with Task 1c best parameters)
KERNEL = "rbf"
C = 1.0
GAMMA = "scale"
CLASS_WEIGHT = "balanced"

def load_data(path):
    df = pd.read_csv(path)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].str.strip().replace("", np.nan)
    return df

def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    ordi = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("enc", OrdinalEncoder(categories=[ORD_ORDER])),
                     ("sc", StandardScaler())])
    nom = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_COLS),
                              ("ord", ordi, ORD_COLS),
                              ("nom", nom, NOM_COLS)])

def main():
    # Take test path from command line arguments
    if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
        test_path = sys.argv[1]
    else:
        test_path = "Customer_test.csv"
        print(f"Running interactively: defaulting test path to '{test_path}'")

    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Training file '{TRAIN_PATH}' not found in the current directory.")
        return
        
    if not os.path.exists(test_path):
        print(f"Error: Test file '{test_path}' not found.")
        return

    # Load datasets
    train = load_data(TRAIN_PATH)
    test = load_data(test_path)

    y = train[TARGET].values
    classes = np.array(sorted(np.unique(y)))

    # Preprocessing learned from training data only
    pre = build_preprocessor()
    X_train = pre.fit_transform(train[NUM_COLS + ORD_COLS + NOM_COLS])
    X_test = pre.transform(test[NUM_COLS + ORD_COLS + NOM_COLS])

    # K binary SVMs: class k (1) vs all others (0)
    scores = np.zeros((X_test.shape[0], len(classes)))
    for k, c in enumerate(classes):
        clf = SVC(kernel=KERNEL, C=C, gamma=GAMMA, class_weight=CLASS_WEIGHT)
        clf.fit(X_train, (y == c).astype(int))
        scores[:, k] = clf.decision_function(X_test)

    # Class with the highest decision score
    pred_idx = np.argmax(scores, axis=1)
    predictions = classes[pred_idx]

    # Save to ova.csv
    output_df = pd.DataFrame({"predicted": predictions})
    output_df.to_csv("ova.csv", index=False)
    print("One-vs-All SVM predictions successfully saved to 'ova.csv'")

if __name__ == "__main__":
    main()
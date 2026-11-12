import sys
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC

# ---------------- Configuration ----------------
# We assume the training file is located in the same directory from which the script is run.
TRAIN_PATH = "Customer_train.csv"
TARGET = "Segmentation"

# Feature Categorization based on the dataset
NUM_COLS = ["Age", "Work_Experience", "Family_Size"]
ORD_COLS = ["Spending_Score"]
ORD_ORDER = ["Low", "Average", "High"]
NOM_COLS = ["Gender", "Ever_Married", "Graduated", "Profession", "Var_1"]

# Binary SVM hyperparameters (placeholder values; to be updated with Task 1c best parameters)
KERNEL = "rbf"
C = 1.0
GAMMA = "scale"
CLASS_WEIGHT = "balanced"  # Handles the class imbalance requirement


def load_data(path):
    """Loads CSV and strips white spaces from string columns."""
    df = pd.read_csv(path)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].str.strip().replace("", np.nan)
    return df


def build_preprocessor():
    """Builds a pipeline handling missing values and scales appropriately per feature type."""
    num_transformer = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    
    ord_transformer = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(categories=[ORD_ORDER])),
        ("sc", StandardScaler())
    ])
    
    nom_transformer = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    return ColumnTransformer([
        ("num", num_transformer, NUM_COLS),
        ("ord", ord_transformer, ORD_COLS),
        ("nom", nom_transformer, NOM_COLS)
    ])


def main():
    # Take test path from command line arguments
    if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
        test_path = sys.argv[1]
    else:
        # Fallback for interactive testing
        test_path = "Customer_test.csv"
        print(f"Running interactively: defaulting test path to '{test_path}'")
        
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Training file '{TRAIN_PATH}' not found in the current directory.")
        return
        
    if not os.path.exists(test_path):
        print(f"Error: Test file '{test_path}' not found.")
        return

    # Load data
    train = load_data(TRAIN_PATH)
    test = load_data(test_path)

    y = train[TARGET].values
    classes = np.array(sorted(np.unique(y)))
    K = len(classes)
    
    # Calculate class priors for tie-breaking
    prior = np.array([(y == c).mean() for c in classes])

    # Learn all preprocessing from training data only
    pre = build_preprocessor()
    X_train = pre.fit_transform(train[NUM_COLS + ORD_COLS + NOM_COLS])
    X_test = pre.transform(test[NUM_COLS + ORD_COLS + NOM_COLS])

    # Initialize votes and scores matrices
    votes = np.zeros((X_test.shape[0], K), dtype=int)
    scores = np.zeros((X_test.shape[0], K))

    # Train (K choose 2) binary SVMs using only the rows of those two classes
    for i, j in combinations(range(K), 2):
        mask = (y == classes[i]) | (y == classes[j])
        X_train_bin = X_train[mask]
        y_train_bin = y[mask]
        
        # 1 means class j, 0 means class i
        y_bin = (y_train_bin == classes[j]).astype(int) 
        
        clf = SVC(kernel=KERNEL, C=C, gamma=GAMMA, class_weight=CLASS_WEIGHT)
        clf.fit(X_train_bin, y_bin)

        # Decision function > 0 favors class j, <= 0 favors class i
        d = clf.decision_function(X_test)
        
        votes[d > 0, j] += 1
        votes[d <= 0, i] += 1
        
        # Accumulate decision values for explicit, order-independent tie-breaking
        scores[:, j] += d
        scores[:, i] -= d

    # Predict using majority voting and tie-breaking
    pred_idx = np.empty(X_test.shape[0], dtype=int)
    for n in range(X_test.shape[0]):
        cand = np.where(votes[n] == votes[n].max())[0]
        if len(cand) > 1:
            # Tie breaking step 1: Highest sum of decision margins
            best = cand[np.isclose(scores[n, cand], scores[n, cand].max())]
            if len(best) > 1:
                # Tie breaking step 2: Largest training class prior
                best = best[prior[best] == prior[best].max()]
            cand = best
        pred_idx[n] = cand[0]

    # Save to ovo.csv with a single column 'predicted'
    output_df = pd.DataFrame({"predicted": classes[pred_idx]})
    output_df.to_csv("ovo.csv", index=False)
    print("One-vs-One SVM predictions successfully saved to 'ovo.csv'")

if __name__ == "__main__":
    main()
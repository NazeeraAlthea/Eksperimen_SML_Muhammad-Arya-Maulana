import kagglehub
import shutil
import os
import pandas as pd

# -------------------------------
# Load dataset
# -------------------------------
def load_dataset():
    cache_path = kagglehub.dataset_download(
        "lainguyn123/student-performance-factors"
    )

    target_path = "student-performance_raw"
    os.makedirs(target_path, exist_ok=True)

    shutil.copytree(cache_path, target_path, dirs_exist_ok=True)

    df = pd.read_csv(
        "student-performance_raw/StudentPerformanceFactors.csv"
    )
    return df


# -------------------------------
# Preprocessing steps
# -------------------------------
def fill_missing_values(df):
    df = df.copy()

    df["Teacher_Quality"] = df["Teacher_Quality"].fillna(
        df["Teacher_Quality"].mode()[0]
    )
    df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
        df["Parental_Education_Level"].mode()[0]
    )
    df["Distance_from_Home"] = df["Distance_from_Home"].fillna(
        df["Distance_from_Home"].mode()[0]
    )
    return df


def create_multiclass_label(df):
    df = df.copy()

    df["Performance_Level"] = pd.qcut(
        df["Exam_Score"],
        q=3,
        labels=["Low", "Medium", "High"]
    )
    return df


def encode_features(df):
    X = df.drop(columns=["Performance_Level", "Exam_Score"])
    y = df["Performance_Level"]

    X_encoded = pd.get_dummies(X, drop_first=True)

    df_final = X_encoded.copy()
    df_final["Performance_Level"] = y.values
    return df_final


# -------------------------------
# Main pipeline
# -------------------------------
def run_preprocessing():
    df = load_dataset()
    df = fill_missing_values(df)
    df = create_multiclass_label(df)
    df_final = encode_features(df)

    os.makedirs(
        "preprocessing/student-performance_preprocessing",
        exist_ok=True
    )

    df_final.to_csv(
        "preprocessing/student-performance_preprocessing/data.csv",
        index=False
    )

    print("Preprocessing selesai. Dataset siap dilatih.")


if __name__ == "__main__":
    run_preprocessing()

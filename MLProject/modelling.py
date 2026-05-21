import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train():
    # Mengambil data dari dalam folder MLProject
    df = pd.read_csv("namadataset_preprocessing/cleaned_ethereum_data.csv")
    X = df.drop(columns=['FLAG'])
    y = df['FLAG']

    # Mulai tracking secara lokal di dalam container
    with mlflow.start_run() as run:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rf.fit(X, y)
        
        # Log model dengan nama "model"
        mlflow.sklearn.log_model(rf, "model")
        
        print(f"✅ Model berhasil dilatih dan di-log dengan Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train():
    df = pd.read_csv("namadataset_preprocessing/cleaned_ethereum_data.csv")
    X = df.drop(columns=['FLAG'])
    y = df['FLAG']

    stabil_env = {
        "name": "ethereum-fraud-env",
        "channels": ["conda-forge", "nodefaults"], 
        "dependencies": [
            "python=3.10.12",  
            "pip",
            {"pip": ["pandas", "numpy", "scikit-learn", "mlflow==2.19.0"]}
        ]
    }

    with mlflow.start_run() as run:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rf.fit(X, y)
        
        # Menyematkan environment yang kebal ToS ke dalam model
        mlflow.sklearn.log_model(rf, "model", conda_env=stabil_env)
        
        print(f"✅ Model berhasil dilatih dan di-log dengan Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()
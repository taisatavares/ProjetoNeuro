from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

app = FastAPI(title="Projeto Neuro API", version="1.0")

# ----------------- PERFORMANCE -----------------
class Record(BaseModel):
    VAR2: str
    IDADE: float
    VAR5: str
    REF_DATE: str
    TARGET: int
    # adicione outras variáveis se precisar

class BatchRecords(BaseModel):
    batch_records: List[Record]

@app.post("/performance")
def calculate_performance(data: BatchRecords):
    try:
        df = pd.DataFrame([r.dict() for r in data.batch_records])
        df.fillna(np.nan, inplace=True)

        # Carregar modelo
        with open(r"C:\Users\taisa\PycharmProjects\ProjetoNeuro\monitoring\model.pkl", "rb") as f:
            model = pickle.load(f)

        X = df.drop(columns=["REF_DATE", "TARGET"])
        y_true = df["TARGET"]

        y_score = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y_true, y_score)

        df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
        volumetria = df.groupby(df["REF_DATE"].dt.to_period("M")).size().to_dict()

        return {"volumetria": volumetria, "roc_auc": roc_auc}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- ADERÊNCIA -----------------
class AdherenceInput(BaseModel):
    dataset_path: str

@app.post("/aderencia")
def calculate_adherence(data: AdherenceInput):
    try:
        df_input = pd.read_csv(data.dataset_path, sep=",")
        df_input.fillna(np.nan, inplace=True)

        with open(r"C:\Users\taisa\PycharmProjects\ProjetoNeuro\monitoring\model.pkl", "rb") as f:
            model = pickle.load(f)

        X_input = df_input.drop(columns=["TARGET"], errors="ignore")
        scores_input = model.predict_proba(X_input)[:, 1]

        df_test = pd.read_csv("datasets/credit_01/test.gz", compression="gzip")
        X_test = df_test.drop(columns=["TARGET"], errors="ignore")
        scores_test = model.predict_proba(X_test)[:, 1]

        ks_stat, ks_pvalue = ks_2samp(scores_input, scores_test)

        return {"ks_statistic": ks_stat, "ks_pvalue": ks_pvalue}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- ROOT -----------------
@app.get("/")
def root():
    return {"message": "API Projeto Neuro está rodando!"}

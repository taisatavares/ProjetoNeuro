from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

router = APIRouter(prefix="/performance", tags=["performance"])

class Record(BaseModel):
    VAR2: str
    IDADE: float
    VAR5: str
    REF_DATE: str
    TARGET: int

class BatchRecords(BaseModel):
    batch_records: List[Record]

@router.post("/")
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


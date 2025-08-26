from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from scipy.stats import ks_2samp

router = APIRouter(prefix="/aderencia", tags=["aderencia"])

class AdherenceInput(BaseModel):
    dataset_path: str

@router.post("/")
def calculate_adherence(data: AdherenceInput):
    try:
        df_input = pd.read_csv(data.dataset_path)
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


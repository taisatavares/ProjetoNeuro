from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# =========================
# Configurações e utilitários
# =========================

# Caminhos padrão (pode sobrescrever por variáveis de ambiente)
MODEL_CANDIDATES = [
    os.environ.get("MONITORING_MODEL_PATH", "../monitoring/model.pkl"),
    "monitoring/model.pkl",
    "model.pkl",
    "best_model.pkl"
]
REF_TEST_PATH = os.environ.get("MONITORING_REF_TEST_PATH", "../datasets/credit_01/test.gz")

NON_FEATURE_COLS = {"TARGET", "REF_DATE", "ID", "ID_COL", "ANO", "MES", "DIA"}

app = FastAPI(title="Credit Monitoring API", version="1.0.0")

_model = None
_model_features = None  # ordem/nomes esperados, se disponíveis


def _load_model():
    global _model, _model_features
    if _model is not None:
        return _model

    last_err = None
    for path in MODEL_CANDIDATES:
        try:
            if os.path.exists(path):
                _model = joblib.load(path)
                break
        except Exception as e:
            last_err = e
    if _model is None:
        raise RuntimeError(
            f"Não foi possível carregar o modelo. Verifique MODEL_CANDIDATES={MODEL_CANDIDATES}. "
            f"Último erro: {last_err}"
        )

    # feature_names_in_ costuma existir em modelos/pipelines do sklearn 1.0+
    _model_features = getattr(_model, "feature_names_in_", None)
    return _model


def _to_dataframe_from_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    # Converte lista de dicts em DataFrame e normaliza nulos
    df = pd.DataFrame(records)

    # Garantir existência de colunas importantes (podem vir ausentes em alguns registros)
    if "REF_DATE" in df.columns:
        # tentar converter data
        df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], errors="coerce", utc=True)
    if "TARGET" in df.columns:
        # normalizar alvo para int/0-1 se vier como string/float
        df["TARGET"] = pd.to_numeric(df["TARGET"], errors="coerce").astype("float")

    # Substituir strings "null" por np.nan, se aparecerem
    df = df.replace({"null": np.nan, "None": np.nan})

    return df


def _align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara X para o modelo:
    - remove colunas sabidamente não-features
    - alinha à ordem esperada se feature_names_in_ existir
    - adiciona colunas ausentes com np.nan
    """
    model = _load_model()

    # Remove colunas não-features comuns (modelo pode ignorar se tiver ColumnTransformer interno)
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    X = df[cols].copy()

    # Se o modelo expõe os nomes esperados, alinhar exatamente:
    if getattr(model, "feature_names_in_", None) is not None:
        expected = list(model.feature_names_in_)
        # adicionar colunas que faltam
        for c in expected:
            if c not in X.columns:
                X[c] = np.nan
        # remover extras
        X = X[expected]

    return X


def _safe_predict_proba(df: pd.DataFrame) -> np.ndarray:
    """
    Retorna p_mau (probabilidade da classe 1).
    """
    model = _load_model()
    X = _align_features(df)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        # binária mas sem 2 colunas por algum motivo:
        # fallback: se só 1 coluna, assumir que já é p(classe1)
        return proba.ravel()
    elif hasattr(model, "decision_function"):
        # fallback: normalizar decision_function para [0,1] (sigmóide)
        from scipy.special import expit
        scores = model.decision_function(X)
        return expit(scores)
    else:
        raise RuntimeError("Modelo não possui predict_proba nem decision_function.")


def _read_dataset_flex(path: str) -> pd.DataFrame:
    """
    Lê dataset local em formatos comuns. Tenta:
    - CSV (sep=',' e sep=';'), com/sem compressão .gz
    - Parquet
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # tentativas
    tries = []

    def try_read(fn, desc):
        tries.append(desc)
        return fn()

    # 1) CSV padrão
    try:
        return try_read(lambda: pd.read_csv(path), "pd.read_csv()")
    except Exception:
        pass

    # 2) CSV ; (muito comum nesses datasets)
    try:
        return try_read(lambda: pd.read_csv(path, sep=";"), "pd.read_csv(sep=';')")
    except Exception:
        pass

    # 3) CSV com compressão inferida
    try:
        return try_read(lambda: pd.read_csv(path, compression="infer"), "pd.read_csv(compression='infer')")
    except Exception:
        pass

    # 4) Parquet
    try:
        return try_read(lambda: pd.read_parquet(path), "pd.read_parquet()")
    except Exception:
        raise RuntimeError(f"Falha ao ler dataset em {path}. Tentativas: {tries}")


# =========================
# Schemas dos endpoints
# =========================
class PerformancePayload(BaseModel):
    records: List[Dict[str, Any]]


class AdherencePayload(BaseModel):
    dataset_path: str


# =========================
# Endpoints
# =========================

@app.get("/health")
def health():
    _load_model()
    return {"status": "ok", "model_loaded": True}


@app.post("/performance")
def performance(payload: PerformancePayload):
    """
    Recebe: {"records": [ {...}, {...}, ... ]}
    Cada registro deve conter REF_DATE (string) e TARGET (0/1), além das variáveis do modelo.
    Retorna: volumetria por mês (REF_DATE) e AUC-ROC no lote.
    """
    try:
        df = _to_dataframe_from_records(payload.records)

        if "REF_DATE" not in df.columns:
            raise HTTPException(status_code=400, detail="Coluna 'REF_DATE' ausente nos registros.")
        if "TARGET" not in df.columns:
            raise HTTPException(status_code=400, detail="Coluna 'TARGET' ausente nos registros.")

        # Volumetria por mês (YYYY-MM)
        df["ref_month"] = df["REF_DATE"].dt.strftime("%Y-%m")
        volumetria = df.groupby("ref_month").size().to_dict()

        # Calcular AUC usando apenas linhas com TARGET válido (0/1)
        df_valid = df.dropna(subset=["TARGET"]).copy()
        # PRECISA ter pelo menos duas classes para AUC
        y = df_valid["TARGET"].astype(int)
        if y.nunique() < 2:
            auc = None
            note = "Não foi possível calcular AUC (TARGET possui apenas uma classe no lote)."
        else:
            p_mau = _safe_predict_proba(df_valid)
            auc = float(roc_auc_score(y, p_mau))
            note = None

        return {
            "volumetria_por_mes": volumetria,
            "auc_roc": auc,
            "n_total_registros": int(df.shape[0]),
            "n_usados_no_auc": int(df_valid.shape[0]),
            "nota": note,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no /performance: {e}")


@app.post("/adherence")
def adherence(payload: AdherencePayload):
    """
    Recebe: {"dataset_path": "caminho/para/base.ext"}
    Lê a base, escora com o mesmo modelo e compara a distribuição de scores
    com a base de Teste (../datasets/credit_01/test.gz por padrão) usando KS.
    """
    try:
        # Ler base fornecida
        df_in = _read_dataset_flex(payload.dataset_path)
        # Normalizar tipos básicos
        if "REF_DATE" in df_in.columns:
            df_in["REF_DATE"] = pd.to_datetime(df_in["REF_DATE"], errors="coerce", utc=True)

        # Escorar base de input
        p_in = _safe_predict_proba(df_in)

        # Ler base de Teste de referência
        df_ref = _read_dataset_flex(REF_TEST_PATH)
        if "REF_DATE" in df_ref.columns:
            df_ref["REF_DATE"] = pd.to_datetime(df_ref["REF_DATE"], errors="coerce", utc=True)
        p_ref = _safe_predict_proba(df_ref)

        # KS entre as duas distribuições de score
        ks_stat, ks_pvalue = ks_2samp(p_in, p_ref)

        return {
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "n_amostra_input": int(len(p_in)),
            "n_amostra_referencia": int(len(p_ref)),
            "referencia_teste_path": REF_TEST_PATH
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no /adherence: {e}")

#!/usr/bin/env python3

import pandas as pd

import numpy as np

from pathlib import Path



from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss



from xgboost import XGBClassifier



DATA_PATH = "/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h_meanminmax_exact.parquet"

OUT_DIR = Path("/nfs/turbo/si-acastel/mimic-project/derived/model_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)



FOLDS_OUT = OUT_DIR / "ehr_xgb_pneumothorax_tuned_folds.csv"

SUMMARY_OUT = OUT_DIR / "ehr_xgb_pneumothorax_tuned_summary.csv"



LABEL_COL = "y_pneumothorax"



# Best params from RandomizedSearchCV

BEST = dict(

    subsample=0.6,

    reg_lambda=0.5,

    n_estimators=1000,

    min_child_weight=1,

    max_depth=8,

    learning_rate=0.05,

    colsample_bytree=0.8,

)



def get_feature_cols(df: pd.DataFrame):

    lab_cols = [c for c in df.columns if c.startswith("lab_")]

    demo_cols = ["anchor_age", "gender"]

    feat_cols = [c for c in (demo_cols + lab_cols) if c in df.columns]

    return feat_cols



def main():

    df = pd.read_parquet(DATA_PATH)

    dft = df[df[LABEL_COL].notna()].copy()

    dft[LABEL_COL] = dft[LABEL_COL].astype(int)



    feat_cols = get_feature_cols(dft)

    X = dft[feat_cols]

    y = dft[LABEL_COL]



    print(f"Rows={len(dft):,} prevalence={y.mean():.4f} features={len(feat_cols)}")



    prep = Pipeline(steps=[

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler()),

    ])



    clf = XGBClassifier(

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=42,

        n_jobs=8,

        **BEST

    )



    pipe = Pipeline(steps=[

        ("prep", prep),

        ("clf", clf),

    ])



    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



    rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):

        X_tr, X_te = X.iloc[tr], X.iloc[te]

        y_tr, y_te = y.iloc[tr], y.iloc[te]



        pipe.fit(X_tr, y_tr)



        p_tr = pipe.predict_proba(X_tr)[:, 1]

        p_te = pipe.predict_proba(X_te)[:, 1]



        rows.append({

            "fold": fold,

            "n_train": int(len(y_tr)),

            "n_test": int(len(y_te)),

            "prevalence": float(y_te.mean()),

            "train_logloss": float(log_loss(y_tr, p_tr, labels=[0,1])),

            "auroc": float(roc_auc_score(y_te, p_te)),

            "auprc": float(average_precision_score(y_te, p_te)),

            "model": "xgb_pneumothorax_tuned",

            "task": "pneumothorax",

        })



    folds = pd.DataFrame(rows)

    folds.to_csv(FOLDS_OUT, index=False)



    summary = folds[["train_logloss","auroc","auprc","prevalence"]].agg(["mean","std"]).reset_index()

    summary = summary.rename(columns={"index":"stat"})

    summary["model"] = "xgb_pneumothorax_tuned"

    summary["task"] = "pneumothorax"

    summary.to_csv(SUMMARY_OUT, index=False)



    print("Wrote:", FOLDS_OUT)

    print("Wrote:", SUMMARY_OUT)

    print(summary)



if __name__ == "__main__":

    main()

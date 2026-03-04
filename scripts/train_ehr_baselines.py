#!/usr/bin/env python3

import pandas as pd

import numpy as np

from pathlib import Path



from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import (

    roc_auc_score,

    average_precision_score,

    log_loss,

)



from xgboost import XGBClassifier





DATA_PATH = "/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h_meanminmax_exact.parquet"

OUT_DIR = "/nfs/turbo/si-acastel/mimic-project/derived/model_results"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)





LABELS = {

    "pneumonia": "y_pneumonia",

    "pneumothorax": "y_pneumothorax",

}



def get_feature_cols(df):

    lab_cols = [c for c in df.columns if c.startswith("lab_")]

    demo_cols = ["anchor_age", "gender"]

    feat_cols = [c for c in (demo_cols + lab_cols) if c in df.columns]

    return feat_cols



def eval_fold(y_true, y_prob):

    y_pred = (y_prob >= 0.5).astype(int)

    return {

        "train_logloss": None,

        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,

        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,

        "prevalence": float(np.mean(y_true)),

    }



def make_xgb(seed=42):
    # CPU baseline
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=8,
     )


def make_logreg():

    # Standard baseline with regularization

    return LogisticRegression(

        solver="lbfgs",

        max_iter=2000,

        class_weight=None,

    )



def build_preprocess(numeric_cols):

    return Pipeline(steps=[

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler(with_mean=True, with_std=True)),

    ])



def run_cv(model_name, model, X, y, numeric_cols, n_splits=5, seed=42):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)



    preprocess = build_preprocess(numeric_cols)

    pipe = Pipeline(steps=[

        ("prep", preprocess),

        ("clf", model),

    ])



    rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]

        y_tr, y_te = y.iloc[tr_idx].astype(int), y.iloc[te_idx].astype(int)



        pipe.fit(X_tr, y_tr)



        # probs

        p_tr = pipe.predict_proba(X_tr)[:, 1]

        p_te = pipe.predict_proba(X_te)[:, 1]



        # metrics

        tr_logloss = float(log_loss(y_tr, p_tr, labels=[0,1]))

        te = eval_fold(y_te, p_te)

        te["train_logloss"] = tr_logloss



        te["model"] = model_name

        te["fold"] = fold

        te["n_train"] = int(len(y_tr))

        te["n_test"] = int(len(y_te))



        rows.append(te)



    dfm = pd.DataFrame(rows)

    summary = dfm[["train_logloss","auroc","auprc","prevalence"]].agg(["mean","std"])

    return dfm, summary



def main():

    df = pd.read_parquet(DATA_PATH)



    feat_cols = get_feature_cols(df)

    print("Feature columns:", len(feat_cols))



    all_outputs = []

    all_summaries = []



    for task, ycol in LABELS.items():

        # Use only rows labeled for that task

        dft = df[df[ycol].notna()].copy()

        dft[ycol] = dft[ycol].astype(int)



        X = dft[feat_cols]

        y = dft[ycol]



        print(f"\nTask={task} rows={len(dft):,} prevalence={y.mean():.4f}")



        # XGBoost

        xgb_df, xgb_sum = run_cv(f"xgb_{task}", make_xgb(), X, y, feat_cols)

        xgb_df["task"] = task

        all_outputs.append(xgb_df)



        xgb_sum["model"] = f"xgb_{task}"

        xgb_sum["task"] = task

        all_summaries.append(xgb_sum.reset_index().rename(columns={"index":"stat"}))



        # Logistic Regression

        lr_df, lr_sum = run_cv(f"logreg_{task}", make_logreg(), X, y, feat_cols)

        lr_df["task"] = task

        all_outputs.append(lr_df)



        lr_sum["model"] = f"logreg_{task}"

        lr_sum["task"] = task

        all_summaries.append(lr_sum.reset_index().rename(columns={"index":"stat"}))



    out_folds = pd.concat(all_outputs, ignore_index=True)

    out_sum = pd.concat(all_summaries, ignore_index=True)



    folds_path = f"{OUT_DIR}/ehr_baseline_folds.csv"

    sum_path = f"{OUT_DIR}/ehr_baseline_summary.csv"

    out_folds.to_csv(folds_path, index=False)

    out_sum.to_csv(sum_path, index=False)



    print("\nWrote:", folds_path)

    print("Wrote:", sum_path)

    print("\nPreview summary:")

    print(out_sum)



if __name__ == "__main__":

    main()

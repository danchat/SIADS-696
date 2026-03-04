#!/usr/bin/env python3

import json

from pathlib import Path



import numpy as np

import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier



DATA_PATH = "/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_labpanel_v2_24h_meanminmax_exact.parquet"

OUT_DIR = Path("/nfs/turbo/si-acastel/mimic-project/derived/model_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)



RESULTS_CSV = OUT_DIR / "xgb_pneumothorax_v2_randomsearch_cv_results.csv"

BEST_JSON = OUT_DIR / "xgb_pneumothorax_v2_randomsearch_best.json"



LABEL_COL = "y_pneumothorax"





def get_feature_cols(df: pd.DataFrame):

    lab_cols = [c for c in df.columns if c.startswith("lab_")]

    demo_cols = ["anchor_age"]

    feat_cols = [c for c in (demo_cols + lab_cols) if c in df.columns]

    return feat_cols





def main():

    print("Loading:", DATA_PATH)

    df = pd.read_parquet(DATA_PATH)



    dft = df[df[LABEL_COL].notna()].copy()

    dft[LABEL_COL] = dft[LABEL_COL].astype(int)



    feat_cols = get_feature_cols(dft)

    X = dft[feat_cols]

    y = dft[LABEL_COL]



    print(f"Task=pneumothorax rows={len(dft):,} prevalence={y.mean():.4f} features={len(feat_cols)}")



    prep = Pipeline(steps=[

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler()),

    ])



    base_model = XGBClassifier(

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=42,

        n_jobs=8,

    )



    pipe = Pipeline(steps=[

        ("prep", prep),

        ("clf", base_model),

    ])



    # Focused search around your best region

    param_dist = {

        "clf__n_estimators": np.arange(800, 2201, 200),          

        "clf__max_depth": np.arange(5, 11),               

        "clf__learning_rate": np.array([0.01, 0.02, 0.03, 0.05, 0.07]),

        "clf__subsample": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),

        "clf__colsample_bytree": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),

        "clf__min_child_weight": np.array([1, 2, 5, 10]),

        "clf__reg_lambda": np.array([0.0, 0.5, 1.0, 2.0, 5.0]),

        "clf__gamma": np.array([0.0, 0.5, 1.0, 2.0]),            

    }



    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



    search = RandomizedSearchCV(

        estimator=pipe,

        param_distributions=param_dist,

        n_iter=60,                         

        scoring="average_precision",       

        cv=cv,

        verbose=2,

        n_jobs=1,                           

        random_state=42,

        return_train_score=True,

    )



    print("Starting RandomizedSearchCV (v2)")

    search.fit(X, y)



    print("\nBest CV AUPRC:", search.best_score_)

    print("Best params:")

    for k, v in search.best_params_.items():

        print(f"  {k}: {v}")



    results_df = pd.DataFrame(search.cv_results_)

    results_df.to_csv(RESULTS_CSV, index=False)

    print("Wrote:", RESULTS_CSV)



    best_payload = {

        "task": "pneumothorax",

        "label_col": LABEL_COL,

        "n_rows": int(len(dft)),

        "prevalence": float(y.mean()),

        "best_cv_auprc": float(search.best_score_),

        "best_params": search.best_params_,

        "data_path": DATA_PATH,

    }

    BEST_JSON.write_text(json.dumps(best_payload, indent=2))

    print("Wrote:", BEST_JSON)





if __name__ == "__main__":

    main()

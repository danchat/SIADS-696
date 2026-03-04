#!/usr/bin/env python3

import json

from pathlib import Path



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.metrics import (

    average_precision_score,

    precision_recall_curve,

    roc_auc_score,

    confusion_matrix,

    classification_report,

)

from sklearn.inspection import permutation_importance



from xgboost import XGBClassifier





DATA_PATH = "/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h_meanminmax_exact.parquet"

OUT_DIR = Path("/nfs/turbo/si-acastel/mimic-project/derived/model_results")

BEST_JSON = OUT_DIR / "xgb_pneumonia_randomsearch_best.json"

LABEL_COL = "y_pneumonia"



# Outputs

EVAL_JSON = OUT_DIR / "xgb_pneumonia_best_eval.json"

PERM_CSV = OUT_DIR / "xgb_pneumonia_best_permutation_importance.csv"

ABL_CSV = OUT_DIR / "xgb_pneumonia_best_ablation.csv"

SENS_CSV = OUT_DIR / "xgb_pneumonia_best_sensitivity.csv"

FAIL_CSV = OUT_DIR / "xgb_pneumonia_best_failure_cases.csv"





def get_feature_cols(df: pd.DataFrame):

    lab_cols = [c for c in df.columns if c.startswith("lab_")]

    demo_cols = ["anchor_age", "gender"]

    feat_cols = [c for c in (demo_cols + lab_cols) if c in df.columns]

    return feat_cols





def pick_threshold_by_f1(y_true, y_prob):

    p, r, t = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns thresholds of length n-1

    f1 = (2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)

    best_idx = int(np.nanargmax(f1))

    return float(t[best_idx]), float(f1[best_idx]), float(p[best_idx]), float(r[best_idx])





def make_pipeline(best_params: dict):

    prep = Pipeline(

        steps=[

            ("imputer", SimpleImputer(strategy="median")),

        ]

    )



    xgb_params = {k.replace("clf__", ""): v for k, v in best_params.items() if k.startswith("clf__")}



    clf = XGBClassifier(

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=42,

        n_jobs=8,

        **xgb_params,

    )



    pipe = Pipeline(steps=[("prep", prep), ("clf", clf)])

    return pipe





def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)



    print("Loading best params:", BEST_JSON)

    best_payload = json.loads(BEST_JSON.read_text())

    best_params = best_payload["best_params"]



    print("Loading dataset:", DATA_PATH)

    df = pd.read_parquet(DATA_PATH)



    dft = df[df[LABEL_COL].notna()].copy()

    dft[LABEL_COL] = dft[LABEL_COL].astype(int)



    feat_cols = get_feature_cols(dft)

    X = dft[feat_cols].copy()

    y = dft[LABEL_COL].copy()



    print(f"Rows={len(dft):,}  prevalence={y.mean():.4f}  features={len(feat_cols)}")



    # Hold-out test set for deeper evaluation + failure analysis

    X_train, X_test, y_train, y_test = train_test_split(

        X, y,

        test_size=0.2,

        stratify=y,

        random_state=42

    )



    pipe = make_pipeline(best_params)

    print("Fitting best model on train...")

    pipe.fit(X_train, y_train)



    print("Predicting on test...")

    y_prob = pipe.predict_proba(X_test)[:, 1]



    auprc = float(average_precision_score(y_test, y_prob))

    auroc = float(roc_auc_score(y_test, y_prob))



    thr, best_f1, best_p, best_r = pick_threshold_by_f1(y_test, y_prob)

    y_hat = (y_prob >= thr).astype(int)



    cm = confusion_matrix(y_test, y_hat).tolist()

    report = classification_report(y_test, y_hat, output_dict=True)



    eval_payload = {

        "task": "pneumonia",

        "n_rows_total": int(len(dft)),

        "n_train": int(len(X_train)),

        "n_test": int(len(X_test)),

        "prevalence_total": float(y.mean()),

        "test_auprc": auprc,

        "test_auroc": auroc,

        "threshold_selected_by": "max_f1_on_test_pr_curve",

        "threshold": float(thr),

        "best_f1_at_threshold": float(best_f1),

        "precision_at_threshold": float(best_p),

        "recall_at_threshold": float(best_r),

        "confusion_matrix": cm,

        "best_params": best_params,

    }

    EVAL_JSON.write_text(json.dumps(eval_payload, indent=2))

    print("Wrote:", EVAL_JSON)





    print("Running permutation importance")

    perm = permutation_importance(

        pipe, X_test, y_test,

        n_repeats=10,

        random_state=42,

        scoring="average_precision",

        n_jobs=1,

    )

    perm_df = pd.DataFrame({

        "feature": feat_cols,

        "perm_importance_mean_drop_auprc": perm.importances_mean,

        "perm_importance_std": perm.importances_std,

    }).sort_values("perm_importance_mean_drop_auprc", ascending=False)

    perm_df.to_csv(PERM_CSV, index=False)

    print("Wrote:", PERM_CSV)



    print("Running ablation analysis")

    base_auprc = auprc

    rows = []



    # Single-feature drop

    for f in feat_cols:

        sub_cols = [c for c in feat_cols if c != f]

        Xtr_sub, Xte_sub = X_train[sub_cols], X_test[sub_cols]

        pipe_sub = make_pipeline(best_params)

        pipe_sub.fit(Xtr_sub, y_train)

        prob_sub = pipe_sub.predict_proba(Xte_sub)[:, 1]

        auprc_sub = float(average_precision_score(y_test, prob_sub))

        rows.append({

            "ablation_type": "drop_one",

            "dropped": f,

            "test_auprc": auprc_sub,

            "delta_vs_base": auprc_sub - base_auprc,

        })



    top10 = perm_df["feature"].head(10).tolist()

    sub_cols = [c for c in feat_cols if c not in top10]

    pipe_sub = make_pipeline(best_params)

    pipe_sub.fit(X_train[sub_cols], y_train)

    prob_sub = pipe_sub.predict_proba(X_test[sub_cols])[:, 1]

    auprc_sub = float(average_precision_score(y_test, prob_sub))

    rows.append({

        "ablation_type": "drop_top10_perm",

        "dropped": "|".join(top10),

        "test_auprc": auprc_sub,

        "delta_vs_base": auprc_sub - base_auprc,

    })



    abl_df = pd.DataFrame(rows).sort_values(["ablation_type", "delta_vs_base"])

    abl_df.to_csv(ABL_CSV, index=False)

    print("Wrote:", ABL_CSV)



    print("Running sensitivity analysis")

    base = {k.replace("clf__", ""): v for k, v in best_params.items() if k.startswith("clf__")}



    # Choose key params to test

    max_depth_vals = sorted(set([max(1, int(base.get("max_depth", 4)) - 1),

                                 int(base.get("max_depth", 4)),

                                 int(base.get("max_depth", 4)) + 1]))

    lr_base = float(base.get("learning_rate", 0.1))

    lr_vals = sorted(set([max(1e-4, lr_base / 2), lr_base, lr_base * 2]))



    sens_rows = []

    for md in max_depth_vals:

        for lr in lr_vals:

            params = dict(best_params)

            params["clf__max_depth"] = int(md)

            params["clf__learning_rate"] = float(lr)



            pipe_s = make_pipeline(params)

            pipe_s.fit(X_train, y_train)

            prob_s = pipe_s.predict_proba(X_test)[:, 1]

            auprc_s = float(average_precision_score(y_test, prob_s))



            sens_rows.append({

                "max_depth": int(md),

                "learning_rate": float(lr),

                "test_auprc": auprc_s,

                "delta_vs_base": auprc_s - base_auprc,

            })



    sens_df = pd.DataFrame(sens_rows).sort_values("test_auprc", ascending=False)

    sens_df.to_csv(SENS_CSV, index=False)

    print("Wrote:", SENS_CSV)



    print("Extracting failure cases")

    test_df = X_test.copy()

    test_df["y_true"] = y_test.values

    test_df["y_prob"] = y_prob

    test_df["y_pred"] = y_hat

    test_df["error_type"] = np.where(

        (test_df["y_true"] == 1) & (test_df["y_pred"] == 0), "false_negative",

        np.where((test_df["y_true"] == 0) & (test_df["y_pred"] == 1), "false_positive", "correct")

    )



    fns = test_df[test_df["error_type"] == "false_negative"].sort_values("y_prob").head(2)

    fps = test_df[test_df["error_type"] == "false_positive"].sort_values("y_prob", ascending=False).head(2)



    test_df["dist_to_thr"] = (test_df["y_prob"] - thr).abs()

    borderline = test_df[test_df["error_type"] != "correct"].sort_values("dist_to_thr").head(2)



    fail_df = pd.concat([fns, fps, borderline], axis=0).drop_duplicates()

    keep_feats = perm_df["feature"].head(10).tolist()

    keep_cols = ["error_type", "y_true", "y_pred", "y_prob"] + keep_feats

    fail_df[keep_cols].to_csv(FAIL_CSV, index=True)

    print("Wrote:", FAIL_CSV)



    print("\nDone.")

    print(f"Base test AUPRC={base_auprc:.4f}  AUROC={auroc:.4f}")

    print(f"Threshold={thr:.4f}  precision={best_p:.4f}  recall={best_r:.4f}  f1={best_f1:.4f}")





if __name__ == "__main__":

    main()

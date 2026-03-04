#!/usr/bin/env python3

import json

from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay



from xgboost import XGBClassifier





DATA_PATH = "/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h_meanminmax_exact.parquet"

OUT_DIR = Path("/nfs/turbo/si-acastel/mimic-project/derived/model_results")



BEST_JSON = OUT_DIR / "xgb_pneumonia_randomsearch_best.json"

EVAL_JSON = OUT_DIR / "xgb_pneumonia_best_eval.json"

PERM_CSV = OUT_DIR / "xgb_pneumonia_best_permutation_importance.csv"

ABL_CSV  = OUT_DIR / "xgb_pneumonia_best_ablation.csv"

SENS_CSV = OUT_DIR / "xgb_pneumonia_best_sensitivity.csv"



LABEL_COL = "y_pneumonia"



FIG_DIR = OUT_DIR / "figures"





def get_feature_cols(df: pd.DataFrame):

    lab_cols = [c for c in df.columns if c.startswith("lab_")]

    demo_cols = ["anchor_age", "gender"]

    return [c for c in (demo_cols + lab_cols) if c in df.columns]





def make_pipeline(best_params: dict):

    prep = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])



    xgb_params = {k.replace("clf__", ""): v for k, v in best_params.items() if k.startswith("clf__")}



    clf = XGBClassifier(

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=42,

        n_jobs=8,

        **xgb_params,

    )

    return Pipeline(steps=[("prep", prep), ("clf", clf)])





def savefig(fig, stem: str):

    png_path = FIG_DIR / f"{stem}.png"

    pdf_path = FIG_DIR / f"{stem}.pdf"

    fig.tight_layout()

    fig.savefig(png_path, dpi=300)

    fig.savefig(pdf_path)

    print("Wrote:", png_path)

    print("Wrote:", pdf_path)





def fig_perm_importance():

    perm = pd.read_csv(PERM_CSV)

    top = perm.sort_values("perm_importance_mean_drop_auprc", ascending=False).head(15)



    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(top["feature"][::-1], top["perm_importance_mean_drop_auprc"][::-1])

    ax.set_title("Permutation importance (Top 15) — mean drop in test AUPRC")

    ax.set_xlabel("Mean drop in AUPRC when permuted")

    ax.set_ylabel("Feature")

    savefig(fig, "fig1_perm_importance_top15")





def fig_ablation_drop_one():

    abl = pd.read_csv(ABL_CSV)

    drop_one = abl[abl["ablation_type"] == "drop_one"].copy()

    top_harm = drop_one.sort_values("delta_vs_base", ascending=True).head(15)



    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(top_harm["dropped"][::-1], top_harm["delta_vs_base"][::-1])

    ax.set_title("Ablation (LOFO) — Top 15 drops in test AUPRC")

    ax.set_xlabel("Δ AUPRC vs base (negative = worse)")

    ax.set_ylabel("Removed feature")

    savefig(fig, "fig2_ablation_lofo_top15_drops")





def fig_sensitivity_heatmap():

    sens = pd.read_csv(SENS_CSV)

    piv = sens.pivot(index="max_depth", columns="learning_rate", values="test_auprc").sort_index()



    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(piv.values, aspect="auto")



    ax.set_title("Sensitivity analysis — test AUPRC across (max_depth, learning_rate)")

    ax.set_xlabel("learning_rate")

    ax.set_ylabel("max_depth")



    ax.set_xticks(np.arange(piv.shape[1]))

    ax.set_xticklabels([str(c) for c in piv.columns])

    ax.set_yticks(np.arange(piv.shape[0]))

    ax.set_yticklabels([str(i) for i in piv.index])



    fig.colorbar(im, ax=ax, label="Test AUPRC")

    savefig(fig, "fig3_sensitivity_heatmap_auprc")





def fig_pr_curve_and_confusion():

    best_payload = json.loads(BEST_JSON.read_text())

    best_params = best_payload["best_params"]



    eval_payload = json.loads(EVAL_JSON.read_text())

    thr = float(eval_payload["threshold"])



    df = pd.read_parquet(DATA_PATH)

    dft = df[df[LABEL_COL].notna()].copy()

    dft[LABEL_COL] = dft[LABEL_COL].astype(int)



    feat_cols = get_feature_cols(dft)

    X = dft[feat_cols].copy()

    y = dft[LABEL_COL].copy()



    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, stratify=y, random_state=42

    )



    pipe = make_pipeline(best_params)

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]



    # PR curve

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    ap = average_precision_score(y_test, y_prob)



    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(recall, precision)

    ax.set_title(f"Precision–Recall curve (test) — AUPRC={ap:.3f}")

    ax.set_xlabel("Recall")

    ax.set_ylabel("Precision")



    # Mark the chosen threshold on the curve

    if thresholds.size > 0:

        idx = int(np.argmin(np.abs(thresholds - thr)))

        ax.scatter(recall[idx], precision[idx], marker="o")

        ax.annotate(f"thr={thr:.3f}", (recall[idx], precision[idx]))



    savefig(fig, "fig4_precision_recall_curve")



    # Confusion matrix at selected threshold

    y_pred = (y_prob >= thr).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)



    fig2, ax2 = plt.subplots(figsize=(5.5, 5))

    disp.plot(ax=ax2, values_format="d")

    ax2.set_title(f"Confusion matrix (test) at thr={thr:.3f}")

    savefig(fig2, "fig5_confusion_matrix")





def main():

    FIG_DIR.mkdir(parents=True, exist_ok=True)



    required = [BEST_JSON, EVAL_JSON, PERM_CSV, ABL_CSV, SENS_CSV]

    missing = [str(p) for p in required if not p.exists()]

    if missing:

        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))



    fig_perm_importance()

    fig_ablation_drop_one()

    fig_sensitivity_heatmap()

    fig_pr_curve_and_confusion()



    print("\nAll figures written to:", FIG_DIR)





if __name__ == "__main__":

    main()

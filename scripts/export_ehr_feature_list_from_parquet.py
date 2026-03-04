#!/usr/bin/env python3

from pathlib import Path

import re

import pandas as pd



FEATURES_PATH = "/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h_meanminmax_exact.parquet"

LABMAP_PATH   = "/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h_labmap.parquet"



OUT_DIR = Path("/nfs/turbo/si-acastel/mimic-project/derived/model_results/appendix")

OUT_DIR.mkdir(parents=True, exist_ok=True)



OUT_LONG   = OUT_DIR / "appendix_ehr_features_long.csv"

OUT_COMPACT = OUT_DIR / "appendix_ehr_features_compact.csv"





LABEL_COLS = {"y_pneumonia", "y_pneumothorax"}

ID_LIKE_COLS = {"subject_id", "hadm_id", "study_id"}





def get_feature_cols(df: pd.DataFrame):

    lab_cols = [c for c in df.columns if c.startswith("lab_")]

    demo_cols = [c for c in ["anchor_age", "gender"] if c in df.columns]

    feat_cols = [c for c in (demo_cols + lab_cols) if c not in LABEL_COLS and c not in ID_LIKE_COLS]

    return feat_cols





def load_labmap(path: str) -> pd.DataFrame | None:

    p = Path(path)

    if not p.exists():

        return None

    lm = pd.read_parquet(p)



    cols_lower = {c.lower(): c for c in lm.columns}

    itemid_col = cols_lower.get("itemid") or cols_lower.get("lab_id") or cols_lower.get("labitemid")

    label_col = cols_lower.get("label") or cols_lower.get("lab_label") or cols_lower.get("labname") or cols_lower.get("name")



    if itemid_col is None:

        return lm



    # Standardize column names for merge

    lm2 = lm.copy()

    lm2 = lm2.rename(columns={itemid_col: "itemid"})

    if label_col is not None:

        lm2 = lm2.rename(columns={label_col: "lab_label"})

    else:

        lm2["lab_label"] = None



    keep = ["itemid", "lab_label"]

    for extra in ["fluid", "category", "loinc_code", "unitname"]:

        if extra in lm2.columns:

            keep.append(extra)



    return lm2[keep].drop_duplicates()





def parse_lab_feature(colname: str):

    """

    Expected patterns:

      lab_51006_mean

      lab_51006_min

      lab_51006_max

    """

    m = re.match(r"^lab_(\d+)_(mean|min|max)$", colname)

    if not m:

        return None, None

    return int(m.group(1)), m.group(2)





def main():

    print("Loading features:", FEATURES_PATH)

    df = pd.read_parquet(FEATURES_PATH)



    feat_cols = get_feature_cols(df)

    print(f"Found {len(feat_cols)} feature columns.")



    rows = []

    for c in feat_cols:

        if c.startswith("lab_"):

            itemid, stat = parse_lab_feature(c)

            rows.append({

                "feature_name": c,

                "feature_type": "lab",

                "itemid": itemid,

                "statistic": stat,

            })

        else:

            rows.append({

                "feature_name": c,

                "feature_type": "demographic",

                "itemid": None,

                "statistic": None,

            })



    long_df = pd.DataFrame(rows)



    labmap = load_labmap(LABMAP_PATH)

    if labmap is not None and "itemid" in labmap.columns:

        long_df = long_df.merge(labmap, on="itemid", how="left")

        print("Joined lab map:", LABMAP_PATH)

    else:

        long_df["lab_label"] = None

        print("Lab map not joined (missing file or missing itemid column).")

        if labmap is not None:

            print("Labmap columns:", list(labmap.columns))



    long_df = long_df.sort_values(["feature_type", "lab_label", "itemid", "statistic", "feature_name"], na_position="last")

    long_df.to_csv(OUT_LONG, index=False)

    print("Wrote:", OUT_LONG)



    lab_only = long_df[long_df["feature_type"] == "lab"].copy()

    if not lab_only.empty:

        compact = (

            lab_only.groupby(["itemid", "lab_label"], dropna=False)["statistic"]

            .apply(lambda s: ", ".join(sorted({x for x in s if isinstance(x, str)})))

            .reset_index()

            .rename(columns={"statistic": "available_statistics"})

            .sort_values(["lab_label", "itemid"], na_position="last")

        )

    else:

        compact = pd.DataFrame(columns=["itemid", "lab_label", "available_statistics"])



    demo_only = long_df[long_df["feature_type"] == "demographic"][["feature_name"]].drop_duplicates()

    if not demo_only.empty:

        demo_only = demo_only.assign(itemid=None, lab_label="(demographic)", available_statistics=demo_only["feature_name"])

        demo_only = demo_only[["itemid", "lab_label", "available_statistics"]]

        compact = pd.concat([compact, demo_only], ignore_index=True)



    compact.to_csv(OUT_COMPACT, index=False)

    print("Wrote:", OUT_COMPACT)



    print("\nPreview (compact):")

    print(compact.head(20).to_string(index=False))





if __name__ == "__main__":

    main()

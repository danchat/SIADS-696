#!/usr/bin/env python3

import argparse

import re

from pathlib import Path



import numpy as np

import pandas as pd





STUDY_LABELS = "/nfs/turbo/si-acastel/mimic-project/derived/study_labels.parquet"

PATIENTS = "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz"

LABITEMS = "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz"

LABEVENTS = "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz"



OUT_DIR = Path("/nfs/turbo/si-acastel/mimic-project/derived")

OUT_DIR.mkdir(parents=True, exist_ok=True)





def safe_name(s: str, max_len: int = 40) -> str:

    s = str(s).strip().lower()

    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")

    return s[:max_len] if len(s) > max_len else s





def build_time_window_index(base: pd.DataFrame, window_hours: int) -> pd.DataFrame:

    dt_col = "study_datetime"

    if dt_col not in base.columns:

        raise ValueError(f"{dt_col} not found in columns: {base.columns.tolist()}")



    base = base.copy()

    base[dt_col] = pd.to_datetime(base[dt_col], errors="coerce")

    if base[dt_col].isna().any():

        n_bad = int(base[dt_col].isna().sum())

        raise ValueError(f"{n_bad} rows have unparseable study_datetime.")



    base["cxr_time"] = base[dt_col]

    base["window_end"] = base["cxr_time"]

    base["window_start"] = base["cxr_time"] - pd.Timedelta(hours=window_hours)

    return base





def select_itemids_by_regex(dlab: pd.DataFrame, patterns: list[str], max_itemids: int) -> pd.DataFrame:

    """

    Select itemids from d_labitems using a single combined regex over the label column.

    Returns table with itemid/label/fluid/category.

    """

    dlab = dlab.copy()

    dlab["label"] = dlab["label"].astype(str)



    combined = "(" + "|".join(patterns) + ")"

    mask = dlab["label"].str.contains(combined, case=False, regex=True, na=False)



    hits = dlab.loc[mask, ["itemid", "label", "fluid", "category"]].drop_duplicates(subset=["itemid"])

    hits = hits.sort_values(["category", "label"], kind="stable")



    if len(hits) > max_itemids:

        hits = hits.head(max_itemids)



    return hits





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--window_hours", type=int, default=24)

    ap.add_argument("--chunksize", type=int, default=2_000_000)

    ap.add_argument("--max_itemids", type=int, default=80, help="cap number of matched lab itemids")

    ap.add_argument(

        "--out_name",

        type=str,

        default="ehr_study_labpanel_v2_24h_meanminmax_exact.parquet",

        help="output parquet filename under derived/"

    )

    args = ap.parse_args()



    print("Loading study_labels")

    base = pd.read_parquet(STUDY_LABELS)

    print("Base rows:", len(base))

    print("Columns:", base.columns.tolist())



    # Add demographics

    print("Loading patients (demographics)")

    pats = pd.read_csv(PATIENTS, compression="gzip", usecols=["subject_id", "gender", "anchor_age"])

    base = base.merge(pats, on="subject_id", how="left")



    # Add window_start/window_end around CXR time

    base = build_time_window_index(base, args.window_hours)



    # Build per-study window table

    study_windows = base[["study_id", "hadm_id", "window_start", "window_end"]].dropna(subset=["hadm_id"]).copy()

    study_windows["study_id"] = study_windows["study_id"].astype(int)

    study_windows["hadm_id"] = study_windows["hadm_id"].astype(int)



    # If multiple rows per study_id exist, keep unique study windows

    study_windows = study_windows.drop_duplicates(subset=["study_id"])

    print(f"Study windows: {len(study_windows):,} studies")



    cohort_hadm = study_windows["hadm_id"].unique()

    cohort_hadm_set = set(int(x) for x in cohort_hadm)

    print(f"Unique admissions in cohort: {len(cohort_hadm_set):,}")



    # Load d_labitems and choose a clinically motivated panel by regex

    print("Loading d_labitems and selecting curated lab panel via regex")

    dlab = pd.read_csv(LABITEMS, compression="gzip")



    # Clinically motivated patterns (CBC + acid/base + severity + basic chem + coag + infection if present)

    patterns = [

        # CBC

        r"\bwbc\b", r"white blood cell", r"\bhgb\b", r"hemoglobin", r"\bhct\b", r"hematocrit",

        r"platelet", r"\bplt\b", r"neutroph", r"lymph", r"monocyte", r"eosinoph", r"basoph",

        # Chem / electrolytes

        r"sodium", r"potassium", r"chloride", r"bicarbonate", r"anion gap", r"creatinine",

        r"urea nitrogen", r"\bbun\b", r"glucose", r"magnesium", r"calcium", r"phosph",

        r"albumin", r"total protein",

        # Tissue injury / LFT-ish

        r"\balt\b", r"\bast\b", r"alkaline phosphatase", r"bilirubin", r"lactate dehydrogenase", r"\bldh\b",

        # Acid-base / oxygenation proxies

        r"\bph\b", r"pco2", r"po2", r"base excess", r"oxygen saturation", r"\bso2\b", r"lactate",

        # Coag / severity

        r"\binr\b", r"\bpt\b", r"\bptt\b", r"fibrinogen",

        # Infection markers

        r"c-reactive", r"\bcrp\b", r"procalcitonin",

    ]



    hits = select_itemids_by_regex(dlab, patterns, args.max_itemids)

    if hits.empty:

        raise RuntimeError("No labs matched the regex patterns. Check d_labitems or patterns list.")



    hits["itemid"] = hits["itemid"].astype(int)

    print(f"Selected itemids: {len(hits)}")

    print(hits.to_string(index=False))



    itemids = hits["itemid"].tolist()

    itemid_to_suffix = {int(r.itemid): safe_name(r.label) for r in hits.itertuples(index=False)}



    # Stream labevents and aggregate by (study_id, itemid) within each study's window

    print("Streaming labevents and aggregating by (study_id, itemid) within window...")

    usecols = ["hadm_id", "itemid", "charttime", "valuenum"]



    # agg[(study_id, itemid)] = [sum, count, min, max]

    agg: dict[tuple[int, int], list[float]] = {}



    chunk_i = 0

    for chunk in pd.read_csv(

        LABEVENTS,

        compression="gzip",

        usecols=usecols,

        chunksize=args.chunksize,

        low_memory=False,

    ):

        chunk_i += 1



        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"])

        if chunk.empty:

            continue



        # Types

        chunk["hadm_id"] = chunk["hadm_id"].astype(int, errors="ignore")

        chunk["itemid"] = chunk["itemid"].astype(int, errors="ignore")



        # Filter to cohort admissions and selected itemids

        chunk = chunk[chunk["hadm_id"].isin(cohort_hadm_set)]

        chunk = chunk[chunk["itemid"].isin(itemids)]

        if chunk.empty:

            continue



        # Parse time

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

        chunk = chunk.dropna(subset=["charttime"])

        if chunk.empty:

            continue



        # Join to only the relevant study windows for the hadm_ids present in this chunk

        hadm_in_chunk = chunk["hadm_id"].unique()

        sw = study_windows[study_windows["hadm_id"].isin(hadm_in_chunk)]

        if sw.empty:

            continue



        # Merge

        merged = chunk.merge(sw, on="hadm_id", how="inner")



        # Filter lab rows to those within each study's window

        merged = merged[(merged["charttime"] >= merged["window_start"]) & (merged["charttime"] <= merged["window_end"])]

        if merged.empty:

            continue



        # Update aggregates

        for r in merged.itertuples(index=False):

            study_id = int(r.study_id)

            itemid = int(r.itemid)

            v = float(r.valuenum)



            key = (study_id, itemid)

            if key not in agg:

                agg[key] = [v, 1.0, v, v] 

            else:

                s, c, mn, mx = agg[key]

                s += v

                c += 1.0

                if v < mn:

                    mn = v

                if v > mx:

                    mx = v

                agg[key] = [s, c, mn, mx]



        if chunk_i % 5 == 0:

            print(f"  processed chunks: {chunk_i}, current (study,item) pairs: {len(agg):,}")



    if not agg:

        raise RuntimeError("No lab rows aggregated.")



    print("Computing exact mean, building wide feature table")



    # Convert agg dict to a long dataframe

    rows = []

    for (study_id, itemid), (s, c, mn, mx) in agg.items():

        mean = s / c if c > 0 else np.nan

        suffix = itemid_to_suffix.get(itemid, str(itemid))

        rows.append({

            "study_id": study_id,

            "itemid": itemid,

            "suffix": suffix,

            "mean": mean,

            "min": mn,

            "max": mx,

        })

    long = pd.DataFrame(rows)



    # Create wide columns

    long["col_mean"] = long.apply(lambda r: f"lab_{int(r.itemid)}_{r.suffix}_mean", axis=1)

    long["col_min"] = long.apply(lambda r: f"lab_{int(r.itemid)}_{r.suffix}_min", axis=1)

    long["col_max"] = long.apply(lambda r: f"lab_{int(r.itemid)}_{r.suffix}_max", axis=1)



    wide_mean = long.pivot_table(index="study_id", columns="col_mean", values="mean", aggfunc="first")

    wide_min = long.pivot_table(index="study_id", columns="col_min", values="min", aggfunc="first")

    wide_max = long.pivot_table(index="study_id", columns="col_max", values="max", aggfunc="first")



    feats = pd.concat([wide_mean, wide_min, wide_max], axis=1).reset_index()



    # Merge back into base by study_id 

    out = base.merge(feats, on="study_id", how="left")



    out_path = OUT_DIR / args.out_name

    print("Writing:", out_path)

    out.to_parquet(out_path, index=False)



    lab_cols = [c for c in out.columns if c.startswith("lab_")]

    print("Done. Shape:", out.shape)

    print("Num lab feature cols:", len(lab_cols))

    print("Example lab cols:", lab_cols[:10])





if __name__ == "__main__":

    main()

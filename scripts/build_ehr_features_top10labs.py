#!/usr/bin/env python3

import pandas as pd

import numpy as np

from collections import Counter

import argparse



DEFAULTS = {

    "study_labels": "/nfs/turbo/si-acastel/mimic-project/derived/study_labels.parquet",

    "patients": "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz",

    "labevents": "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz",

    "d_labitems": "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz",

    "out": "/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h.parquet",

}



def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--study_labels", default=DEFAULTS["study_labels"])

    ap.add_argument("--patients", default=DEFAULTS["patients"])

    ap.add_argument("--labevents", default=DEFAULTS["labevents"])

    ap.add_argument("--d_labitems", default=DEFAULTS["d_labitems"])

    ap.add_argument("--out", default=DEFAULTS["out"])

    ap.add_argument("--window_hours", type=float, default=24.0)

    ap.add_argument("--top_k", type=int, default=10)

    ap.add_argument("--chunksize", type=int, default=2_000_000)

    ap.add_argument("--max_studies", type=int, default=0, help="0=all; otherwise sample this many studies for a faster test run")

    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()



def main():

    args = parse_args()



    # Load base cohort with hadm_id + study_datetime + labels

    base = pd.read_parquet(args.study_labels)

    base["subject_id"] = base["subject_id"].astype(int)

    base["hadm_id"] = base["hadm_id"].astype(int)

    base["study_id"] = base["study_id"].astype(int)

    base["study_datetime"] = pd.to_datetime(base["study_datetime"], errors="coerce")

    base = base.dropna(subset=["study_datetime"]).copy()



    if args.max_studies and args.max_studies > 0:

        base = base.sample(n=min(args.max_studies, len(base)), random_state=args.seed)



    hadm_ids = set(base["hadm_id"].unique())

    print(f"Loaded study cohort: {len(base):,} studies, {len(hadm_ids):,} unique admissions")



    # Add demographics

    pats = pd.read_csv(args.patients, compression="gzip", usecols=["subject_id", "gender", "anchor_age"])

    pats["subject_id"] = pats["subject_id"].astype(int)

    base = base.merge(pats, on="subject_id", how="left")



    # Standardize gender to 0/1

    # base["gender"] = base["gender"].map({"M": 1, "F": 0})



    # Pass 1: Find top-K most frequent itemids in labevents within this cohort

    print("Pass 1: counting lab item frequencies in cohort admissions")

    item_counter = Counter()



    usecols = ["hadm_id", "itemid", "valuenum"]

    for chunk_idx, chunk in enumerate(pd.read_csv(args.labevents, compression="gzip", usecols=usecols, chunksize=args.chunksize), start=1):

        chunk = chunk.dropna(subset=["hadm_id", "itemid", "valuenum"]).copy()

        chunk["hadm_id"] = chunk["hadm_id"].astype(int)

        chunk["itemid"] = chunk["itemid"].astype(int)



        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]

        if chunk.empty:

            continue



        item_counter.update(chunk["itemid"].value_counts().to_dict())

        if chunk_idx % 5 == 0:

            print(f"  processed chunks: {chunk_idx}, current distinct itemids: {len(item_counter):,}")



    top_itemids = [itemid for itemid, _ in item_counter.most_common(args.top_k)]

    print(f"Top {args.top_k} itemids: {top_itemids}")



    # Map to human-readable lab names

    dlab = pd.read_csv(args.d_labitems, compression="gzip")

    dlab["itemid"] = dlab["itemid"].astype(int)

    top_map = dlab[dlab["itemid"].isin(top_itemids)][["itemid", "label", "fluid", "category"]].drop_duplicates()

    print("\nTop lab itemids mapped via d_labitems:")

    print(top_map.sort_values("itemid").to_string(index=False))



    # Pass 2: For each (study_id, itemid), keep MOST RECENT lab within [study_datetime-24h, study_datetime]

    print("\nPass 2: extracting most recent lab values within window before CXR time")

    window = pd.Timedelta(hours=float(args.window_hours))



    # Small join table: hadm_id -> (study_id, study_datetime)

    join_tbl = base[["hadm_id", "study_id", "study_datetime"]].copy()



    # dict: (study_id, itemid) -> (charttime, valuenum)

    best = {}



    usecols2 = ["hadm_id", "itemid", "charttime", "valuenum"]

    for chunk_idx, chunk in enumerate(pd.read_csv(args.labevents, compression="gzip", usecols=usecols2, chunksize=args.chunksize), start=1):

        chunk = chunk.dropna(subset=["hadm_id","itemid","charttime","valuenum"]).copy()

        chunk["hadm_id"] = chunk["hadm_id"].astype(int)

        chunk["itemid"] = chunk["itemid"].astype(int)

        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]

        chunk = chunk[chunk["itemid"].isin(top_itemids)]

        if chunk.empty:

            continue



        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

        chunk = chunk.dropna(subset=["charttime"])

        if chunk.empty:

            continue



        merged = chunk.merge(join_tbl, on="hadm_id", how="inner")

        merged["window_start"] = merged["study_datetime"] - window

        merged = merged[(merged["charttime"] >= merged["window_start"]) & (merged["charttime"] <= merged["study_datetime"])]

        if merged.empty:

            continue



        # Iterate rows

        for row in merged.itertuples(index=False):

            # row fields: hadm_id, itemid, charttime, valuenum, study_id, study_datetime, window_start

            key = (int(row.study_id), int(row.itemid))

            ct = row.charttime.to_datetime64()

            val = float(row.valuenum)



            prev = best.get(key)

            if prev is None or ct > prev[0]:

                best[key] = (ct, val)



        if chunk_idx % 5 == 0:

            print(f"  processed chunks: {chunk_idx}, current filled (study,item) pairs: {len(best):,}")



    # Convert dict to wide feature table

    if not best:

        raise RuntimeError("No lab values found in the specified window. Consider increasing the window or verifying timestamps.")



    rows = [(sid, itemid, pd.Timestamp(ct), val) for (sid, itemid), (ct, val) in best.items()]

    lab_long = pd.DataFrame(rows, columns=["study_id", "itemid", "charttime_last", "valuenum_last"])



    lab_wide = lab_long.pivot(index="study_id", columns="itemid", values="valuenum_last").reset_index()

    lab_wide.columns = ["study_id"] + [f"lab_{c}" for c in lab_wide.columns[1:]]



    # Merge features back into base

    out = base.merge(lab_wide, on="study_id", how="left")



    # Add helpful metadata: label names for columns

    map_out = args.out.replace(".parquet", "_labmap.parquet")

    top_map.to_parquet(map_out, index=False)



    out.to_parquet(args.out, index=False)



    print("\nWrote dataset:", args.out)

    print("Wrote lab map:", map_out)

    print("Final rows:", len(out))

    feat_cols = [c for c in out.columns if c.startswith("lab_")]

    print("Lab feature cols:", len(feat_cols))

    print("Example feature columns:", feat_cols[:5])



    # Prevalence sanity (per labeled subset)

    if "y_pneumonia" in out.columns:

        print("Pneumonia prevalence (labeled rows):", float(out["y_pneumonia"].mean(skipna=True)))

    if "y_pneumothorax" in out.columns:

        print("Pneumothorax prevalence (labeled rows):", float(out["y_pneumothorax"].mean(skipna=True)))



if __name__ == "__main__":

    main()

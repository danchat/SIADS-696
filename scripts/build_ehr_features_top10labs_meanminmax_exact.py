#!/usr/bin/env python3

import pandas as pd

import numpy as np

import argparse



STUDY_LABELS = "/nfs/turbo/si-acastel/mimic-project/derived/study_labels.parquet"

PATIENTS = "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz"

LABEVENTS = "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz"



# Top-10 from log

TOP_LABS = [50983, 50971, 50902, 50912, 51006, 50882, 50868, 50931, 51221, 50960]



def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--window_hours", type=float, default=24.0)

    ap.add_argument("--chunksize", type=int, default=2_000_000)

    ap.add_argument(

        "--out",

        default="/nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h_meanminmax_exact.parquet"

    )

    return ap.parse_args()



def main():

    args = parse_args()

    window = pd.Timedelta(hours=float(args.window_hours))



    print("Loading study_labels")

    base = pd.read_parquet(STUDY_LABELS)

    base["subject_id"] = base["subject_id"].astype(int)

    base["hadm_id"] = base["hadm_id"].astype(int)

    base["study_id"] = base["study_id"].astype(int)

    base["study_datetime"] = pd.to_datetime(base["study_datetime"], errors="coerce")

    base = base.dropna(subset=["study_datetime"]).copy()

    print("Base rows:", len(base))



    print("Loading patients (demographics)")

    pats = pd.read_csv(PATIENTS, compression="gzip", usecols=["subject_id", "gender", "anchor_age"])

    pats["subject_id"] = pats["subject_id"].astype(int)

    base = base.merge(pats, on="subject_id", how="left")



    # Join table

    join_tbl = base[["hadm_id", "study_id", "study_datetime"]].copy()

    hadm_ids = set(join_tbl["hadm_id"].unique())

    print("Unique admissions in cohort:", len(hadm_ids))



    # Columns: sum, count, min, max

    running = None



    usecols = ["hadm_id", "itemid", "charttime", "valuenum"]

    print("Streaming labevents")

    for i, chunk in enumerate(

        pd.read_csv(LABEVENTS, compression="gzip", usecols=usecols, chunksize=args.chunksize),

        start=1

    ):

        chunk = chunk.dropna(subset=["hadm_id","itemid","charttime","valuenum"]).copy()

        chunk["hadm_id"] = chunk["hadm_id"].astype(int)

        chunk["itemid"] = chunk["itemid"].astype(int)



        # Filter admissions + itemids early (big speedup)

        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]

        chunk = chunk[chunk["itemid"].isin(TOP_LABS)]

        if chunk.empty:

            continue



        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

        chunk = chunk.dropna(subset=["charttime"])

        if chunk.empty:

            continue



        merged = chunk.merge(join_tbl, on="hadm_id", how="inner")

        merged["window_start"] = merged["study_datetime"] - window

        merged = merged[

            (merged["charttime"] >= merged["window_start"]) &

            (merged["charttime"] <= merged["study_datetime"])

        ]

        if merged.empty:

            continue



        # Aggregate raw values within this chunk

        agg = merged.groupby(["study_id","itemid"])["valuenum"].agg(

            sum="sum",

            count="count",

            min="min",

            max="max"

        ).reset_index()



        if running is None:

            running = agg

        else:

            # Combine with running by adding sum/count and min/min, max/max

            running = running.merge(

                agg, on=["study_id","itemid"], how="outer", suffixes=("_r","_c")

            )



            # Fill missing pieces with 0 or +/- inf appropriately

            running["sum_r"] = running["sum_r"].fillna(0.0)

            running["count_r"] = running["count_r"].fillna(0)

            running["min_r"] = running["min_r"].fillna(np.inf)

            running["max_r"] = running["max_r"].fillna(-np.inf)



            running["sum_c"] = running["sum_c"].fillna(0.0)

            running["count_c"] = running["count_c"].fillna(0)

            running["min_c"] = running["min_c"].fillna(np.inf)

            running["max_c"] = running["max_c"].fillna(-np.inf)



            running["sum"] = running["sum_r"] + running["sum_c"]

            running["count"] = running["count_r"] + running["count_c"]

            running["min"] = np.minimum(running["min_r"], running["min_c"])

            running["max"] = np.maximum(running["max_r"], running["max_c"])



            running = running[["study_id","itemid","sum","count","min","max"]]



        if i % 5 == 0:

            n_pairs = 0 if running is None else len(running)

            print(f"  processed chunks: {i}, current (study,item) pairs: {n_pairs:,}")



    if running is None or running.empty:

        raise RuntimeError("No lab values found after filtering. Try increasing window_hours or verify timestamps.")



    print("Computing exact mean")

    running["mean"] = running["sum"] / running["count"].replace(0, np.nan)



    # Wide pivot: lab_<itemid>_<stat>

    print("Pivoting wide...")

    wide = running.pivot(index="study_id", columns="itemid", values=["mean","min","max"])

    wide.columns = [f"lab_{itemid}_{stat}" for stat, itemid in wide.columns]

    wide = wide.reset_index()



    print("Merging features back into base")

    out = base.merge(wide, on="study_id", how="left")



    # Encode gender (keep NaN if missing)

    out["gender"] = out["gender"].map({"M": 1, "F": 0})



    print("Writing:", args.out)

    out.to_parquet(args.out, index=False)



    feat_cols = [c for c in out.columns if c.startswith("lab_")]

    print("Done. Shape:", out.shape)

    print("Num lab features:", len(feat_cols))

    print("Example features:", feat_cols[:6])



if __name__ == "__main__":

    main()

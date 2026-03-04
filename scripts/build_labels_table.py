#!/usr/bin/env python3

import pandas as pd

import numpy as np



STUDY_TO_HADM = "/nfs/turbo/si-acastel/mimic-project/derived/study_to_hadm.parquet"

CHEXPERT = "/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0_gcs/mimic-cxr-2.0.0-chexpert.csv.gz"

OUT = "/nfs/turbo/si-acastel/mimic-project/derived/study_labels.parquet"



def to_binary_label(s: pd.Series) -> pd.Series:

    """

    CheXpert: 1.0, 0.0, -1.0, NaN

    Baseline: keep only {0,1}; treat -1/NaN as missing

    """

    s = pd.to_numeric(s, errors="coerce")

    return s.where(s.isin([0.0, 1.0]))



def main():

    anchor = pd.read_parquet(STUDY_TO_HADM)

    anchor["subject_id"] = anchor["subject_id"].astype(int)

    anchor["study_id"] = anchor["study_id"].astype(int)



    chex = pd.read_csv(CHEXPERT, compression="gzip", usecols=["subject_id","study_id","Pneumonia","Pneumothorax"])

    chex["subject_id"] = chex["subject_id"].astype(int)

    chex["study_id"] = chex["study_id"].astype(int)



    chex["y_pneumonia"] = to_binary_label(chex["Pneumonia"])

    chex["y_pneumothorax"] = to_binary_label(chex["Pneumothorax"])



    df = anchor.merge(

        chex[["subject_id","study_id","y_pneumonia","y_pneumothorax"]],

        on=["subject_id","study_id"],

        how="inner"

    )



    # Keep rows that have at least one usable label

    df = df[~(df["y_pneumonia"].isna() & df["y_pneumothorax"].isna())].copy()



    df.to_parquet(OUT, index=False)



    print("Wrote:", OUT)

    print("Rows:", len(df))

    print("Pneumonia labeled rows:", df["y_pneumonia"].notna().sum())

    print("Pneumothorax labeled rows:", df["y_pneumothorax"].notna().sum())

    print("Pneumonia prevalence:", float(df["y_pneumonia"].mean(skipna=True)))

    print("Pneumothorax prevalence:", float(df["y_pneumothorax"].mean(skipna=True)))



if __name__ == "__main__":

    main()

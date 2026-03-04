#!/usr/bin/env python3



import pandas as pd

import numpy as np





ADMISSIONS_PATH = "/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"

CXR_METADATA_PATH = "/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0_gcs/mimic-cxr-2.0.0-metadata.csv.gz"

OUTPUT_PATH = "/nfs/turbo/si-acastel/mimic-project/derived/study_to_hadm.parquet"





def main():



    print("Loading admissions")

    admissions = pd.read_csv(

        ADMISSIONS_PATH,

        compression="gzip",

        usecols=["subject_id", "hadm_id", "admittime", "dischtime"]

    )



    admissions["subject_id"] = admissions["subject_id"].astype(int)

    admissions["hadm_id"] = admissions["hadm_id"].astype(int)

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")

    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")



    print("Admissions rows:", len(admissions))



    print("Loading CXR metadata")

    cxr = pd.read_csv(

        CXR_METADATA_PATH,

        compression="gzip",

        usecols=["subject_id", "study_id", "StudyDate", "StudyTime"]

    )



    cxr["subject_id"] = cxr["subject_id"].astype(int)

    cxr["study_id"] = cxr["study_id"].astype(int)



    # Combine StudyDate and StudyTime into datetime

    cxr["StudyDate"] = cxr["StudyDate"].astype(str).str.zfill(8)

    cxr["StudyTime"] = cxr["StudyTime"].astype(str)



    cxr["study_datetime"] = pd.to_datetime(

        cxr["StudyDate"] + " " + cxr["StudyTime"],

        format="%Y%m%d %H%M%S.%f",

        errors="coerce"

    )



    cxr = cxr.dropna(subset=["study_datetime"])



    print("CXR rows:", len(cxr))



    print("Merging on subject_id")

    merged = cxr.merge(admissions, on="subject_id", how="inner")



    print("After subject merge:", len(merged))



    print("Filtering by time overlap...")

    merged = merged[

        (merged["study_datetime"] >= merged["admittime"]) &

        (merged["study_datetime"] <= merged["dischtime"])

    ].copy()



    print("After time filter:", len(merged))



    # If multiple matches per study, keep closest admission start

    merged["time_diff"] = (merged["study_datetime"] - merged["admittime"]).abs()

    merged = merged.sort_values(["study_id", "time_diff"])

    merged = merged.drop_duplicates(subset=["study_id"], keep="first")



    print("Final linked rows:", len(merged))



    final = merged[[

        "subject_id",

        "study_id",

        "hadm_id",

        "study_datetime"

    ]].copy()



    print("Saving to:", OUTPUT_PATH)

    final.to_parquet(OUTPUT_PATH, index=False)



    print("Done.")

    print("Output rows:", len(final))





if __name__ == "__main__":

    main()

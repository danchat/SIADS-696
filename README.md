# SIADS 696 Final Project

This repository contains the code for our **SIADS 696 Milestone II project**, which investigates machine learning methods for predicting clinical outcomes using both **radiological imaging data and structured electronic health record (EHR) data**.

---

## Repository Contents

This repository includes:

- **Project Report**
  - `SIADS696_Final_Project_Report.pdf`

- **Source Code**
  - Python scripts and Jupyter Notebooks used for:
    - Data preprocessing
    - Feature engineering
    - Model training
    - Hyperparameter tuning
    - Model evaluation and analysis

Due to data usage restrictions, no project data files are stored in this repository.

---

## Data Access

The datasets used in this project are subject to PhysioNet data use agreements and cannot be redistributed through GitHub.

During development, the datasets were securely stored and accessed using the University of Michigan Great Lakes HPC cluster on Turbo storage. Project paths referenced in the code correspond to the structure of our data on Turbo, found here: https://greatlakes.arc-ts.umich.edu/pun/sys/dashboard/files/fs//nfs/turbo/si-acastel/mimic-project 

---

## External Data Sources

The following datasets were used in this project. These datasets are publicly available but require credentialed access through PhysioNet.

### MIMIC-IV (Structured EHR Data)

https://physionet.org/content/mimiciv/3.1/

MIMIC-IV is a database of de-identified electronic health records. In this project, laboratory measurements and demographic variables were extracted from MIMIC-IV to construct structured EHR features used in supervised machine learning models.

### MIMIC-CXR-JPG (Chest X-ray Images)

https://physionet.org/content/mimic-cxr-jpg/2.1.0/

MIMIC-CXR-JPG contains chest radiographs derived from the MIMIC-CXR database. These images were used to train convolutional neural networks for pneumonia and pneumothorax detection.

Access to both datasets requires completion of PhysioNet credentialing and acceptance of the associated data use agreements.
---

## Computing Environment

Experiments were conducted using two computing environments:

- **University of Michigan Great Lakes HPC Cluster**  
  Used for data storage, preprocessing, and large-scale model training.

- **Vocareum Shared Development Environment**  
  Used for collaborative experimentation and code development.

All file paths in the code reflect the Turbo storage directory structure used on Great Lakes.
---

## Important Notes

- This repository contains all the source code used in the project. Some files may not be relevant to the final models discussed in the report. 

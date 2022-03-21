import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime as dt
##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
    """
    :param icd9_object: ICD-9 code (Pandas/Numpy object).
    :return: extracted main digits of ICD-9 code
    """
    icd9_str = str(icd9_object)
    # TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
    # TODO: Read the homework description carefully.
    if icd9_str[0] == 'E':
        converted = icd9_str[0:4]
    elif icd9_str[0] == 'V':
        converted = icd9_str[0:3]
    else:
        converted = icd9_str[0:3]
    return converted


def build_codemap(df_icd9, transform):
    """
    :return: Dict of code map {main-digits of ICD9: unique feature ID}
    """
    # TODO: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
    df_digits = df_icd9['ICD9_CODE'].apply(transform)
    df_digits_ls = df_digits.tolist()
    df_icd9_ls = df_icd9['ICD9_CODE'].tolist()
    codemap = dict(zip(df_digits_ls, df_icd9_ls))
    return codemap


def create_dataset(path, codemap, transform):
    """
    :param path: path to the directory contains raw files.
    :param codemap: 3-digit ICD-9 code feature map
    :param transform: e.g. convert_icd9
    :return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
    """
    # TODO: 1. Load data from the three csv files
    # TODO: Loading the mortality file is shown as an example below. Load two other files also.
    df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
    df_admissions = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
    df_diagnoses_icd = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))

    # TODO: 2. Convert diagnosis code in to unique feature ID.
    # TODO: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
    df_diagnoses_icd['ICD9_CODE'] = df_diagnoses_icd['ICD9_CODE'].apply(transform)
    df_diagnoses_icd['ICD9_CODE_TRANSFORM'] = LabelEncoder().fit_transform(df_diagnoses_icd['ICD9_CODE'])
    labelmap = dict(zip(df_diagnoses_icd['ICD9_CODE'].tolist(), df_diagnoses_icd['ICD9_CODE_TRANSFORM'].tolist()))

    # TODO: 3. Group the diagnosis codes for the same visit.
    icd_group = df_diagnoses_icd.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE_TRANSFORM'].apply(list).reset_index()

    # TODO: 4. Group the visits for the same patient.
    icd_time_group = pd.merge(icd_group, df_admissions[['HADM_ID', 'ADMITTIME']], on='HADM_ID', how='left')
    icd_time_group['ADMITTIME'] = pd.to_datetime(icd_time_group['ADMITTIME'])

    # TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
    # TODO: Visits for each patient must be sorted in chronological order.
    icd_time_group.index = icd_time_group['ADMITTIME']
    icd_time_group = icd_time_group.sort_index()
    subject_group = icd_time_group.groupby(['SUBJECT_ID'])['ICD9_CODE_TRANSFORM'].apply(list).reset_index()
    subject_group = pd.merge(subject_group, df_mortality, on='SUBJECT_ID', how='left')

    # TODO: 6. Make patient-id List and label List also.
    # TODO: The order of patients in the three List output must be consistent.
    patient_ids = subject_group['SUBJECT_ID'].tolist()
    labels = subject_group['MORTALITY'].tolist()
    seq_data = subject_group['ICD9_CODE_TRANSFORM'].tolist()
    return patient_ids, labels, seq_data


def main():
    # Build a code map from the train set
    print("Build feature id map")
    df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
    codemap = build_codemap(df_icd9, convert_icd9)
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Train set
    print("Construct train set")
    train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

    pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Validation set
    print("Construct validation set")
    validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

    pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb') ,pickle.HIGHEST_PROTOCOL)
    pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Test set
    print("Construct test set")
    test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

    pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

    print("Complete!")


if __name__ == '__main__':
    main()

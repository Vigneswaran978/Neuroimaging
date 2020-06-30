''' Kaggle Competion - TReNDS Neuroimaging'''

# Data preparation method 1

import pandas as pd

MAIN_DATA_PATH = 'D:/neuro/'

# The first set of features are source-based morphometry (SBM) loadings. These are subject-level weights from
# a group-level ICA decomposition of gray matter concentration maps from structural MRI (sMRI) scans.

loading_df = pd.read_csv(
    MAIN_DATA_PATH + 'loading.csv')  # loading.csv - sMRI SBM loadings for both train and test samples

print(loading_df.shape)
print(loading_df.shape[0] / 2)

# The second set are static functional network connectivity (FNC) matrices. These are the subject-level cross-correlation
# values among 53 component timecourses estimated from GIG-ICA of resting state functional MRI (fMRI).

fnc_df = pd.read_csv(
    MAIN_DATA_PATH + 'fnc.csv')  # fnc.csv - static FNC correlation features for both train and test samples

print(fnc_df.shape)

# The third set of features are the component spatial maps (SM). These are the subject-level 3D images of 53 spatial networks
# estimated from GIG-ICA of resting state functional MRI (fMRI).

icn_numbers_df = pd.read_csv(MAIN_DATA_PATH + 'ICN_numbers.csv')

# ICN_numbers.csv - intrinsic connectivity network numbers for each fMRI spatial map; matches FNC names

print(icn_numbers_df.shape)

# train_scores.csv - age and assessment values for train samples

train_scores_df = pd.read_csv(MAIN_DATA_PATH + 'train_scores.csv')

train_scores_df.isna().sum()

train_scores_df['is_train_set'] = 'yes'

print(train_scores_df.shape)

# fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")

df = df.merge(train_scores_df, on="Id", how="left")

train_df = df[df["is_train_set"] == 'yes']

print(df.isnull().sum())
print(train_df.isnull().sum())

train_df = train_df.drop(['is_train_set', 'Id'], axis=1)
train_df.to_csv("train_df1.csv")

# train_df = train_df.dropna()

test_df = df[df['is_train_set'] != 'yes']
test_df.to_csv("test_df1.csv")




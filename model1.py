''' Kaggle Competion - TReNDS Neuroimaging'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras

from sklearn.model_selection import train_test_split
er
from sklearn.metrics import mean_absolute_error
from sklearn.impute import KNNImputer

# useful functions

def plot_mape(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('mean_absolute_percentage_error')
    plt.plot(hist['epoch'], hist['mean_absolute_percentage_error'],
             label='Train MeanAbsolutePercentageError')
    plt.plot(hist['epoch'], hist['val_mean_absolute_percentage_error'],
             label='Val mean_absolute_percentage_error')
    plt.legend()
    plt.show()


def plot_acc(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],
             label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'],
             label='Val accuracy')
    plt.legend()
    plt.show()


# Func to check MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


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

# # Features that are sensitive to Site
# dim_red = ['CON(88)_vs_CON(63)',
#    'IC_07',
#    'IC_05',
#    'IC_16',
#    'IC_26',
#    'IC_06',
#    'IC_18',
#    'IC_12',
#    'IC_24',
#    'IC_15',
#    'IC_13',
#    'IC_02',
#    'IC_08',
#    'IC_21',
#    'IC_28',
#    'IC_20',
#    'IC_30',
#    'IC_22',
#    'IC_29',
#    'IC_14']
#



train_df = df[df["is_train_set"] == 'yes']

df.isnull().sum()
train_df.isnull().sum()


train_ids = train_df[['Id']]
train_ids = train_ids.reset_index()

train_df = train_df.drop(['is_train_set', 'Id'], axis=1)

train_col = train_df.columns

train_col

# train_df = train_df.dropna()
train_df.isnull().sum()

imputer = KNNImputer(n_neighbors=5)
train_df = imputer.fit_transform(train_df)

train_df = pd.DataFrame(train_df)
train_df.columns = train_col

train_df['Id'] = train_ids.Id

train_df.isnull().sum()

# Columns that are sensitive to site


# Preparing actual test set

target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

test_df = df[df['is_train_set'] != 'yes']
test_df.to_csv("test_df.csv")
test_ids = test_df[["Id"]]
test_ids = test_ids.reset_index()

actual_test = test_df.drop(target_cols, axis=1)
actual_test = actual_test.drop(['Id', 'is_train_set'], axis=1)
actual_test = actual_test.drop(dim_red, axis=1)

X = train_df.drop(target_cols, axis=1)
X = X.drop('Id', axis=1)
X = X.drop(dim_red, axis=1)



y = train_df[['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']]

Y_train = train_df[target_cols]

y1 = np.array(y.age, dtype='float64')
y2 = np.array(y.domain1_var1, dtype='float64')
y3 = np.array(y.domain1_var2, dtype='float64')
y4 = np.array(y.domain2_var1, dtype='float64')
y5 = np.array(y.domain2_var2, dtype='float64')

actual_test.to_csv("actual_test.csv")
X.to_csv("X.csv")
Y_train.to_csv("y_train.csv")


X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.1)

model1 = keras.Sequential()
model1.add(keras.layers.Dense(units=2048, activation='relu', input_shape=[X1_train.shape[1]]))
model1.add(keras.layers.Dropout(rate=0.1))

model1.add(keras.layers.Dense(units=1024, activation="relu"))
model1.add(keras.layers.Dropout(rate=0.15))

model1.add(keras.layers.Dense(units=512, activation="relu"))
model1.add(keras.layers.Dropout(rate=0.05))

model1.add(keras.layers.Dense(units = 1))


model1.compile(optimizer=keras.optimizers.Adam(0.0001), loss='MeanAbsolutePercentageError',
               metrics=['MeanAbsolutePercentageError'])

BATCH_SIZE = 128

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)

history1 = model1.fit(
    x=X1_train,
    y=y1_train,
    shuffle=True,
    epochs=100,
    validation_split=0.05,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]

)

plot_mape(history1)
plt.show()

# model1.save("model1.h5")
y1_pred = model1.predict(X1_test)

y1_pred.shape
age = pd.DataFrame(y1_pred)
y1_pred = age.median(axis=1)
a = pd.DataFrame(y1_test)
b = pd.DataFrame(y1_pred)
a['predict'] = b
a['abs_err'] = abs(a[0] - a.predict)

print(f'mean_absolute_percentage_error {mean_absolute_percentage_error(y1_test, y1_pred)}')

mean_absolute_error(y1_test, y1_pred)


#model1.save("model1.h5")
# id columns in submissin data set
sub1 = test_ids.copy()

# actual test data

age = model1.predict(actual_test)

age = pd.DataFrame(age)
age = pd.DataFrame(age.median(axis=1))
age = age.rename(columns={0: 'age'})

# updating submission dataset with predicted "age" value
sub1['age'] = age['age']
#print(sub1)
print(sub1.head())
# ==============================================================================


# Testing Best Model 2 - domain1_var1

X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.05)

model2 = keras.Sequential()
model2.add(keras.layers.Dense(units=1024, activation='relu', input_shape=[X2_train.shape[1]]))
model2.add(keras.layers.Dropout(rate=0.15))

model2.add(keras.layers.Dense(units=512, activation="relu"))
model2.add(keras.layers.Dropout(rate=0.1))

model2.add(keras.layers.Dense(units=208, activation="relu"))
model2.add(keras.layers.Dropout(rate=0.15))

model2.add(keras.layers.Dense(units=128, activation="relu"))
model2.add(keras.layers.Dropout(rate=0.15))

model2.compile(optimizer=keras.optimizers.Adam(0.0001), loss='MeanAbsolutePercentageError',
               metrics=['MeanAbsolutePercentageError'])

BATCH_SIZE = 128

early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', mode='min', patience=5)

history2 = model2.fit(
    x=X2_train,
    y=y2_train,
    shuffle=True,
    epochs=100,
    validation_split=0.05,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

plot_mape(history2)
plt.show()

y2_pred = model2.predict(X2_test)
y2_pred = pd.DataFrame(y2_pred)
y2_pred = y2_pred.median(axis=1)
y2_pred = pd.DataFrame(y2_pred)

c = pd.DataFrame(y2_test)
c['predict'] = y2_pred

c['abs_err'] = abs(c[0] - c.predict)
print(f'mean_absolute_percentage_error {mean_absolute_percentage_error(y2_test, y2_pred)}')
mean_absolute_error(y2_test, y2_pred)

# actual test data

domain1_var1 = model2.predict(actual_test)
domain1_var1 = pd.DataFrame(domain1_var1)
domain1_var1 = domain1_var1.median(axis=1)
domain1_var1 = pd.DataFrame(domain1_var1)

domain1_var1 = domain1_var1.rename(columns={0: 'domain1_var1'})

# updating submission dataset with domain1_var1 predicted values
sub1['domain1_var1'] = domain1_var1.domain1_var1

sub1

# ===============================================================================

# Testing Best Model 3 - domain1_var2

X3_train, X3_test, y3_train, y3_test = train_test_split(X, y3, test_size=0.05)

model3 = keras.Sequential()
model3.add(keras.layers.Dense(units=2048, activation='relu', input_shape=[X3_train.shape[1]]))
model3.add(keras.layers.Dropout(rate=0.3))

model3.add(keras.layers.Dense(units=1024, activation="relu"))
model3.add(keras.layers.Dropout(rate=0.3))

model3.add(keras.layers.Dense(units=512, activation="relu"))
model3.add(keras.layers.Dropout(rate=0.05))

model3.add(keras.layers.Dense(units=64, activation="relu"))
model3.add(keras.layers.Dropout(rate=0.1))

model3.add(keras.layers.Dense(units=64, activation="relu"))

model3.compile(optimizer=keras.optimizers.Adam(0.0001), loss='MeanAbsolutePercentageError',
               metrics=['MeanAbsolutePercentageError'])

# BATCH_SIZE = 256
early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', mode='min', patience=5)

history3 = model3.fit(
    x=X3_train,
    y=y3_train,
    shuffle=True,
    epochs=100,
    validation_split=0.05,

    callbacks=[early_stop]
)

# batch_size=BATCH_SIZE,

plot_mape(history3)
plt.show()

y3_pred = model3.predict(X3_test)
y3_pred = pd.DataFrame(y3_pred)
y3_pred = pd.DataFrame(y3_pred.median(axis=1))

print(f'MeanAbsolutePercentageError {mean_absolute_percentage_error(y3_test, y3_pred)}')
mean_absolute_error(y3_test, y3_pred)

# Model prodced better results

model3.save("model3-mape-19.h5")

domain1_var2 = model3.predict(actual_test)
domain1_var2 = pd.DataFrame(domain1_var2)
domain1_var2 = pd.DataFrame(domain1_var2.median(axis=1))
domain1_var2 = domain1_var2.rename(columns={0: 'domain1_var2'})

# updating submissiondata set with domain1_var2

sub1['domain1_var2'] = domain1_var2.domain1_var2
sub1

# ----==========================================================================

# Model 4 domain2_var1 prediction
# Model 4 domain2_var1 prediction

X4_train, X4_test, y4_train, y4_test = train_test_split(X, y4, test_size=0.05)

model4 = keras.Sequential()
model4.add(keras.layers.Dense(units=2048, activation='relu', input_shape=[X4_train.shape[1]]))
model4.add(keras.layers.Dropout(rate=0.3))

model4.add(keras.layers.Dense(units=512, activation="relu"))
model4.add(keras.layers.Dropout(rate=0.5))

model4.add(keras.layers.Dense(units=512, activation="relu"))
model4.add(keras.layers.Dropout(rate=0.1))
model4.add(keras.layers.Dense(units=200, activation="relu"))
model4.add(keras.layers.Dropout(rate=0.1))

model4.add(keras.layers.Dense(units=312, activation="relu"))
model4.add(keras.layers.Dropout(rate = 0.30))


model4.add(keras.layers.Dense(units = 288, activation = "relu"))


model4.compile(optimizer=keras.optimizers.Adam(0.0001), loss='MeanAbsolutePercentageError',
               metrics=['MeanAbsolutePercentageError'])

BATCH_SIZE = 256

early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', mode='min', patience=5)

history4 = model4.fit(
    x=X4_train,
    y=y4_train,
    shuffle=True,
    epochs=100,
    validation_split=0.05,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

#

plot_mape(history4)
plt.show()

y4_pred = model4.predict(X4_test)

y4_pred = pd.DataFrame(y4_pred)
y4_pred = y4_pred.median(axis=1)
print(f'mean_absolute_percentage_error {mean_absolute_percentage_error(y4_test, y4_pred)}')
mean_absolute_error(y4_test, y4_pred)

domain2_var1 = model4.predict(actual_test)
domain2_var1 = pd.DataFrame(domain2_var1)
domain2_var1 = pd.DataFrame(domain2_var1.median(axis=1))
domain2_var1 = domain2_var1.rename(columns={0: 'domain2_var1'})

sub1['domain2_var1'] = domain2_var1.domain2_var1

sub1

# Model 5 domain2_var2 prediction


X5_train, X5_test, y5_train, y5_test = train_test_split(X, y5, test_size=0.05)

model5 = keras.Sequential()
model5.add(keras.layers.Dense(units=2048, activation='relu', input_shape=[X5_train.shape[1]]))
model5.add(keras.layers.Dropout(rate=0.5))

model5.add(keras.layers.Dense(units=1024, activation="relu"))
model5.add(keras.layers.Dropout(rate=0.3))

model5.add(keras.layers.Dense(units=512, activation="relu"))
model5.add(keras.layers.Dropout(rate=0.15))

model5.add(keras.layers.Dense(units=248, activation="relu"))

model5.add(keras.layers.Dense(units=304, activation="relu"))

model5.compile(optimizer=keras.optimizers.Adam(0.0001), loss='MeanAbsolutePercentageError',
               metrics=['MeanAbsolutePercentageError'])

BATCH_SIZE = 64

early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', mode='min', patience=5)

history5 = model5.fit(
    x=X5_train,
    y=y5_train,
    shuffle=True,
    epochs=100,
    validation_split=0.05,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

plot_mape(history5)
plt.show()

y5_pred = model5.predict(X5_test)
y5_pred = pd.DataFrame(y5_pred)
y5_pred = y5_pred.median(axis=1)

print(f'mean_absolute_percentage_error {mean_absolute_percentage_error(y5_test, y5_pred)}')
mean_absolute_error(y5_test, y5_pred)

domain2_var2 = model5.predict(actual_test)
domain2_var2 = pd.DataFrame(domain2_var2)
domain2_var2 = pd.DataFrame(domain2_var2.median(axis=1))
domain2_var2 = domain2_var2.rename(columns={0: 'domain2_var2'})

sub1['domain2_var2'] = domain2_var2.domain2_var2

sub1

melt_df = sub1.reset_index().melt(id_vars=['Id'],
                                  value_vars=['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'],
                                  value_name='Predicted')

melt_df['Id'] = melt_df['Id'].astype('str') + '_' + melt_df['variable']

melt_df.to_csv(MAIN_DATA_PATH + "submission20.csv", index=False, columns=['Id', 'Predicted'])




























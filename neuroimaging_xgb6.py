import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


MAIN_DATA_PATH = 'D:/neuro/'
# features that are sensitive to site (block) in the design

train_df = pd.read_csv("train_df1.csv")
train_df = train_df.drop('Unnamed: 0', axis = 1)

rde_col= ['CON(88)_vs_CON(63)',
               'IC_07',
               'IC_05',
               'IC_16',
               'IC_26',
               'IC_06',
               'IC_18',
               'IC_12',
               'IC_24',
               'IC_15',
               'IC_13',
               'IC_02',
               'IC_08',
               'IC_21',
               'IC_28',
               'IC_20',
               'IC_30',
               'IC_22',
               'IC_29',
               'IC_14']

train_df = train_df.drop(rde_col, axis = 1)

target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

X = train_df.drop(target_cols, axis = 1)
y_train = train_df[target_cols]
test_df = pd.read_csv("test_df.csv")
test_ids = test_df[["Id"]]
sub1 = test_ids.reset_index()
actual_test = test_df.drop(target_cols, axis=1)
actual_test = actual_test.drop(['Id', 'is_train_set'], axis=1)
actual_test = actual_test.drop('Unnamed: 0', axis = 1)
actual_test = actual_test.drop(rde_col, axis = 1)



y1 = np.array(y_train.age, dtype='float64')
# y2 = np.array(y_train.domain1_var1, dtype='float64')
# y3 = np.array(y_train.domain1_var2, dtype='float64')
# y4 = np.array(y_train.domain2_var1, dtype='float64')
# y5 = np.array(y_train.domain2_var2, dtype='float64')

#X = X.drop(rde_dm111, axis=1)
#actual_test = actual_test.drop(rde_dm111, axis=1)

rng = np.random.RandomState(31337)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.05)

kf = KFold(n_splits=2, shuffle=True)

for train_index, test_index in kf.split(X1_train):
    xgb_model1 = xgb.XGBRegressor(max_depth=2, n_estimators=45).fit(X1_train, y1_train)
    y1_pred = xgb_model1.predict(X1_test)
    print(mean_absolute_error(y1_test, y1_pred))
    print(mean_absolute_percentage_error(y1_test, y1_pred))

age = xgb_model1.predict((actual_test))
age = pd.DataFrame(age)
age = age.rename(columns={0: 'age'})

# updating submission dataset with predicted "age" value
sub1['age'] = age['age']

# Predicting domain1_var1

target_cols1 = ['age', 'domain1_var2', 'domain2_var1', 'domain2_var2']
xd1v1 = train_df.drop(target_cols1, axis = 1)
xd1v1 = xd1v1.dropna()
y2 = xd1v1[['domain1_var1']]
xd1v1 = xd1v1.drop('domain1_var1', axis = 1)
X2_train, X2_test, y2_train, y2_test = train_test_split(xd1v1, y2, test_size=0.05)

kf = KFold(n_splits=3, shuffle=True)

for train_index, test_index in kf.split(X2_train):
    xgb_model2 = xgb.XGBRegressor(max_depth=2, n_estimators=45).fit(X2_train, y2_train)
    y2_pred = xgb_model2.predict(X2_test)
    print(mean_absolute_error(y2_test, y2_pred))
    print(mean_absolute_percentage_error(y2_test, y2_pred))

xgb_model2.save_model("xgb_model2")

domain1_var1 = xgb_model2.predict(actual_test)
domain1_var1 = pd.DataFrame(domain1_var1)
domain1_var1 = pd.DataFrame(domain1_var1)

domain1_var1 = domain1_var1.rename(columns={0: 'domain1_var1'})

# updating submission dataset with domain1_var1 predicted values
sub1['domain1_var1'] = domain1_var1.domain1_var1

# Predicting domain1_var2


target_cols2 = ['age', 'domain1_var1', 'domain2_var1', 'domain2_var2']
xd1v2 = train_df.drop(target_cols2, axis = 1)
xd1v2 = xd1v2.dropna()
y3 = xd1v2[['domain1_var2']]
xd1v2 = xd1v2.drop('domain1_var2', axis = 1)


X3_train, X3_test, y3_train, y3_test = train_test_split(xd1v2, y3, test_size=0.05)

kf = KFold(n_splits=3, shuffle=True)

for train_index, test_index in kf.split(X3_train):
    xgb_model3 = xgb.XGBRegressor(max_depth=2, n_estimators=30).fit(X3_train, y3_train)
    y3_pred = xgb_model3.predict(X3_test)
    print(mean_absolute_error(y3_test, y3_pred))
    print(mean_absolute_percentage_error(y3_test, y3_pred))


domain1_var2 = xgb_model3.predict(actual_test)
domain1_var2 = pd.DataFrame(domain1_var2)
domain1_var2 = domain1_var2.rename(columns={0: 'domain1_var2'})

# updating submission data set with domain1_var2
sub1['domain1_var2'] = domain1_var2.domain1_var2

# Predicting domain2_var1

target_cols3 = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var2']
xd2v1 = train_df.drop(target_cols3, axis = 1)
xd2v1 = xd2v1.dropna()
y4 = xd2v1[['domain2_var1']]
xd2v1 = xd2v1.drop('domain2_var1', axis = 1)



X4_train, X4_test, y4_train, y4_test = train_test_split(xd2v1, y4, test_size=0.05)

kf = KFold(n_splits=2, shuffle=True)

for train_index, test_index in kf.split(X4_train):
    xgb_model4 = xgb.XGBRegressor(max_depth=3, n_estimators=70, reg_alpha=.25).fit(X4_train, y4_train)
    y4_pred = xgb_model4.predict(X4_test)
    print(mean_absolute_error(y4_test, y4_pred))
    print(mean_absolute_percentage_error(y4_test, y4_pred))


domain2_var1 = xgb_model4.predict(actual_test)
domain2_var1 = pd.DataFrame(domain2_var1)
domain2_var1 = domain2_var1.rename(columns={0: 'domain2_var1'})

sub1['domain2_var1'] = domain2_var1.domain2_var1

# predicting doamin2_var2


target_cols4 = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1']
xd2v2 = train_df.drop(target_cols4, axis = 1)
xd2v2 = xd2v2.dropna()
y5 = xd2v2[['domain2_var2']]
xd2v2 = xd2v2.drop('domain2_var2', axis = 1)




X5_train, X5_test, y5_train, y5_test = train_test_split(xd2v2, y5, test_size=0.05)

kf = KFold(n_splits=3, shuffle=True)

for train_index, test_index in kf.split(X5_train):
    xgb_model5 = xgb.XGBRegressor(max_depth=2, n_estimators=80).fit(X5_train, y5_train)
    y5_pred = xgb_model5.predict(X5_test)
    print(mean_absolute_error(y5_test, y5_pred))
    print(mean_absolute_percentage_error(y5_test, y5_pred))

domain2_var2 = xgb_model5.predict(actual_test)
domain2_var2 = pd.DataFrame(domain2_var2)
domain2_var2 = domain2_var2.rename(columns={0: 'domain2_var2'})

sub1['domain2_var2'] = domain2_var2.domain2_var2


MAIN_DATA_PATH = 'D:/neuro/'

melt_df = sub1.reset_index().melt(id_vars=['Id'],
                                  value_vars=['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'],
                                  value_name='Predicted')

melt_df['Id'] = melt_df['Id'].astype('str') + '_' + melt_df['variable']

melt_df.to_csv(MAIN_DATA_PATH + "submissionxgb10.csv", index=False, columns=['Id', 'Predicted'])

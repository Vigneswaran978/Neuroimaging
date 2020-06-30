
import numpy as np
import pandas as pd
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


X = pd.read_csv("X.csv")
y_train = pd.read_csv("y_train.csv")
actual_test = pd.read_csv("actual_test.csv")

y1 = np.array(y_train.age, dtype='float64')
y2 = np.array(y_train.domain1_var1, dtype='float64')
y3 = np.array(y_train.domain1_var2, dtype='float64')
y4 = np.array(y_train.domain2_var1, dtype='float64')
y5 = np.array(y_train.domain2_var2, dtype='float64')


X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.05)


def model_sel():

    model1 = keras.Sequential()
    model1.add(keras.layers.Dense(units=2048, activation='relu', input_shape=[X1_train.shape[1]]))
    model1.add(keras.layers.Dropout(rate=0.5))

    model1.add(keras.layers.Dense(units=1024, activation="relu"))
    model1.add(keras.layers.Dropout(rate=0.30))

    model1.add(keras.layers.Dense(units=512, activation="relu"))
    model1.add(keras.layers.Dropout(rate=0.5))

    model1.add(keras.layers.Dense(units=432, activation="relu"))
    model1.add(keras.layers.Dropout(rate=0.5))

    model1.add(keras.layers.Dense(units=1, activation="relu"))

    model1.compile(optimizer=keras.optimizers.Adam(0.0001), loss='MeanAbsolutePercentageError',
                   metrics=['MeanAbsolutePercentageError'])
    return model1



BATCH_SIZE = 512

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)


sfs1 = SFS(model,
           k_features=20,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

feature_names = X1_train.columns
sfs1 = sfs1.fit(X1_train, y1_train, custom_feature_names=feature_names)


from xgboost import XGBClassifier
# fit model no training data
model = XGBClassifier()
model.fit(X1_train, y1_train)

y1_pred = model.predict(X1_test)



y1_pred.shape
age = pd.DataFrame(y1_pred)
y1_pred = age.median(axis=1)
a = pd.DataFrame(y1_test)
b = pd.DataFrame(y1_pred)
a['predict'] = b
a['abs_err'] = abs(a[0] - a.predict)

print(f'mean_absolute_percentage_error {mean_absolute_percentage_error(y1_test, y1_pred)}')

mean_absolute_error(y1_test, y1_pred)
predictions = [round(value) for value in y1_pred]


history1 = model_sel.fit(
    x=X1_train,
    y=y1_train,
    shuffle=True,
    epochs=100,
    validation_split=0.05,

    callbacks=[early_stop]

)

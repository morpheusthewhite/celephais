import warnings
# clear stdout from FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
# remove "using XXX backend" from stderr
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
sys.stderr = stderr

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import shuffle

PARTITION_PROPORTION_TRAIN = 0.9


# define base model
def baseline_model(inputs=9):
    # create model
    model = Sequential()
    model.add(Dense(inputs, input_dim=inputs, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

    return model


class StudentsEstimator:
    def __init__(self, training_data):
        """
        :param training_data: the dataset (unencoded) which will be used for train/test
        :param partitionate_dataset: if True divides the given data into training and test, which will
        be used when called .train() and .test(), if False all dataset will be used for training
        """
        if training_data == []:
            print("No data available, exiting train")
            return

        # shuffle data before encoding it
        df_unshuffled = pandas.DataFrame(training_data)
        df = shuffle(df_unshuffled)

        self.Y = df["students"]
        X_unencoded = df.drop("students", axis=1)

        # creating encoders for next encodings
        self.label_encoder_subjects = LabelEncoder()
        self.one_hot_encoder_subjects = OneHotEncoder(sparse=False)
        self.label_encoder_days = LabelEncoder()
        self.one_hot_encoder_days = OneHotEncoder(sparse=False)

        self.X = self.transform_data(X_unencoded, fit=True)

        self.estimator = KerasRegressor(build_fn=baseline_model, inputs=len(self.X.columns), epochs=100,
                                        batch_size=5, verbose=1)

    def transform_data(self, data_frame, fit=False):

        # encode subjects
        subjects_unencoded = data_frame["subject"]

        if fit:
            int_label_subjects = self.label_encoder_subjects.fit_transform(subjects_unencoded).reshape(-1, 1)
            subjects_frame = pandas.DataFrame(self.one_hot_encoder_subjects.fit_transform(int_label_subjects))
        else:
            int_label_subjects = self.label_encoder_subjects.transform(subjects_unencoded).reshape(-1, 1)
            subjects_frame = pandas.DataFrame(self.one_hot_encoder_subjects.transform(int_label_subjects))

        X_subjects_encoded = pandas.concat([subjects_frame, data_frame.drop("subject", axis=1)], axis=1)

        # encode days
        days_unencoded = data_frame["day"]

        if fit:
            int_label_days = self.label_encoder_days.fit_transform(days_unencoded).reshape(-1, 1)
            days_frame = pandas.DataFrame(self.one_hot_encoder_days.fit_transform(int_label_days))
        else:
            int_label_days = self.label_encoder_days.transform(days_unencoded).reshape(-1, 1)
            days_frame = pandas.DataFrame(self.one_hot_encoder_days.transform(int_label_days))

        return pandas.concat([days_frame, X_subjects_encoded.drop("day", axis=1)], axis=1)

    def train(self):
        # train model
        self.estimator.fit(self.X, self.Y, validation_split=0.10)

    def predict(self, prediction_data):
        dp = pandas.DataFrame(prediction_data)

        X = self.transform_data(dp)
        predictions_float = self.estimator.predict(X)

        n_predictions = len(X.index)
        if n_predictions == 1:
            # replace the single element with a list of 1 element
            predictions_float = [predictions_float]

        predictions_int = []
        # casting all predictions to int
        for prediction_float in predictions_float:
            predictions_int.append(int(prediction_float))

        return predictions_int

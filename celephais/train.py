import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# define base model
def baseline_model(inputs=9):
    # create model
    model = Sequential()
    model.add(Dense(inputs, input_dim=inputs, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


class StudentsEstimator:
    def __init__(self, training_data):
        if training_data == []:
            print("No data available, exiting train")
            return

        df = pandas.DataFrame(training_data)

        self.Y_train = df["students"]
        X_unencoded = df.drop("students", axis=1)

        # encode subjects
        subjects_unencoded = X_unencoded["subject"]

        # creating encoders for next encodings
        self.label_encoder_subjects = LabelEncoder()
        self.one_hot_encoder_subjects = OneHotEncoder(sparse=False)
        int_label_subjects = self.label_encoder_subjects.fit_transform(subjects_unencoded).reshape(-1, 1)
        subjects_frame = pandas.DataFrame(self.one_hot_encoder_subjects.fit_transform(int_label_subjects))

        X_subjects_encoded = pandas.concat([subjects_frame, X_unencoded.drop("subject", axis=1)], axis=1)

        # encode days
        days_unencoded = df["day"]

        # creating encoders for next encodings
        self.label_encoder_days = LabelEncoder()
        self.one_hot_encoder_days = OneHotEncoder(sparse=False)
        int_label_days = self.label_encoder_days.fit_transform(days_unencoded).reshape(-1, 1)
        days_frame = pandas.DataFrame(self.one_hot_encoder_days.fit_transform(int_label_days))

        self.X_train = pandas.concat([days_frame, X_subjects_encoded.drop("day", axis=1)], axis=1)

        self.estimator = KerasRegressor(build_fn=baseline_model, inputs=len(self.X_train.columns), epochs=100,
                         batch_size=5, verbose=0)

    def train(self):
        # training model
        self.estimator.fit(self.X_train, self.Y_train)

    def predict(self, prediction_data):
        dp = pandas.DataFrame(prediction_data)

        # encode subjects
        subjects_unencoded = dp["subject"]

        int_label_subjects = self.label_encoder_subjects.transform(subjects_unencoded).reshape(-1, 1)
        subjects_frame = pandas.DataFrame(self.one_hot_encoder_subjects.transform(int_label_subjects))

        X_subjects_encoded = pandas.concat([subjects_frame, dp.drop("subject", axis=1)], axis=1)

        # encode days
        days_unencoded = dp["day"]

        int_label_days = self.label_encoder_days.transform(days_unencoded).reshape(-1, 1)
        days_frame = pandas.DataFrame(self.one_hot_encoder_days.transform(int_label_days))

        X = pandas.concat([days_frame, X_subjects_encoded.drop("day", axis=1)], axis=1)

        return self.estimator.predict(X)

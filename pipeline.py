from dowload_data import create_data_frame
from pandas import get_dummies
from sklearn.pipeline import Pipeline


class Model(object):
    def __init__(self, train_data, train_label, valid_data, valid_label):
        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.train_data = self.clear_data(self.train_data)
        self.valid_data = self.clear_data(self.valid_data)

    def clear_data(self, data):
        # fill nan values in 'Age' and 'Fare' columns
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        # change 'Sex' columns into int type
        data = get_dummies(data, columns=['Sex'])
        print(data.head())
        return data


_train_data, _train_label, _valid_data, _valid_label, _test_data = create_data_frame()
my_model = Model(_train_data, _train_label, _valid_data, _valid_label)
# pipeline = Pipeline()

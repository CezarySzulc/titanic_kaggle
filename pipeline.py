from dowload_data import create_data_frame
from pandas import get_dummies, qcut
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re
import numpy as np


class Model(object):
    def __init__(self, train_data, train_label, valid_data, valid_label, test_data):
        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.test_data = test_data
        self.train_data = self.clear_data(self.train_data)
        self.valid_data = self.clear_data(self.valid_data)
        self.test_data = self.clear_data(self.test_data)

    def clear_data(self, data):
        # fill nan values in 'Age' and 'Fare' columns
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        data['Age'] = qcut(data.Age, q=6, labels=False)
        data['Fare'] = qcut(data.Fare, q=6, labels=False)
        # change 'Sex' columns into int type
        data = get_dummies(data, columns=['Sex'], drop_first=True)
        data = get_dummies(data, columns=['Embarked'], drop_first=True)
        # categorical 'Name' column
        data['Title'] = data['Name'].apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
        data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss'})
        data['Title'] = data['Title'].replace([
            'Don', 'Dona', 'Rev', 'Dr', 'Major', 'Lady', 'Sir',
            'Col', 'Capt', 'Countess', 'Jonkheer'
            ],
            'Special'
        )
        data = get_dummies(data, columns=['Title'], drop_first=True)
        # has cabin
        data['Has_Cabin'] = ~data['Cabin'].isnull()
        # drop unused values
        data.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        return data

    def visualization(self):
        sns.set()
        sns.countplot(x=self.train_label)
        plt.figure()
        sns.countplot(x='Sex_male', data=self.train_data)
        sns.factorplot(x='Survived', col='Sex_male', kind='count', data=self.train_data)
        sns.factorplot(x='Survived', col='Pclass', kind='count', data=self.train_data)
        plt.figure()
        sns.distplot(self.train_data['Age'])
        plt.show()

    def pipeline(self):
        columns = self.test_data.columns.tolist()
        columns.remove('PassengerId')
        model = RandomForestClassifier(
            # n_estimators=100,
            # max_depth=10,
            # n_jobs=-1,
            # random_state=43
        )
        param_grid = {
            'n_estimators': np.arange(5, 100, 5),
            'max_depth': np.arange(1, 20)
        }
        model = GridSearchCV(model, param_grid=param_grid, cv=5)
        model.fit(self.train_data[columns], self.train_label)
        # print(self.test_data.Age)
        print(model.score(self.valid_data[columns], self.valid_label))
        print(classification_report(model.predict(self.valid_data[columns]), self.valid_label))
        self.test_data['Survived'] = model.predict(self.test_data[columns])

    def save_test_output(self, name):
        self.test_data[['PassengerId', 'Survived']].to_csv('data/{}.csv'.format(name), index=False)


_train_data, _train_label, _valid_data, _valid_label, _test_data = create_data_frame()
my_model = Model(_train_data, _train_label, _valid_data, _valid_label, _test_data)
my_model.pipeline()
my_model.save_test_output('random_forest')
# pipeline = Pipeline()
# pipeline = Pipeline()

from dowload_data import create_data_frame
from pandas import get_dummies, qcut
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re
import numpy as np
# linear model
from sklearn.linear_model import LogisticRegressionCV
# xgb
import xgboost as xgb


class Model(object):
    def __init__(self, need_valid_data=False):
        if need_valid_data:
            train_data, train_label, valid_data, valid_label, test_data = create_data_frame(True)
            self.valid_data = valid_data
            self.valid_label = valid_label
            self.valid_data = self.clear_data(self.valid_data)
        else:
            train_data, train_label, test_data = create_data_frame()

        self.need_valid_data = need_valid_data
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.train_data = self.clear_data(self.train_data)
        self.test_data = self.clear_data(self.test_data)

    def clear_data(self, data):
        # fill nan values in 'Age' and 'Fare' columns
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        data['Age'] = qcut(data.Age, q=6, labels=False)
        data['Fare'] = qcut(data.Fare, q=10, labels=False)
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
        # xgboost
        model = xgb.XGBClassifier()
        # linear regression
        # model = LogisticRegressionCV(cv=5)
        # Randorm forest tree clf
        # model = RandomForestClassifier(
        #     random_state=43,
        #     n_jobs=-1,
        #     max_depth=6,
        #     n_estimators=40,
        #
        # )
        # param_grid = {
        #     # 'n_estimators': np.arange(5, 100, 5),
        #     # 'max_depth': np.arange(1, 12),
        #     'min_samples_split': np.arange(2, 8)
        # }
        # model = GridSearchCV(model, param_grid=param_grid, cv=5)
        model.fit(self.train_data[columns], self.train_label)
        # print('Model best params: {}'.format(model.best_params_))
        # print('Model best score: {}'.format(model.best_score_))
        # print(model.best_estimator_)
        # print(model.scores_)
        if self.need_valid_data:
            print(model.score(self.valid_data[columns], self.valid_label))
            print(classification_report(model.predict(self.valid_data[columns]), self.valid_label))
        self.test_data['Survived'] = model.predict(self.test_data[columns])

    def save_test_output(self, name):
        self.test_data[['PassengerId', 'Survived']].to_csv('data/{}.csv'.format(name), index=False)


my_model = Model()
my_model.pipeline()
my_model.save_test_output('xgboost')
# pipeline = Pipeline()
# pipeline = Pipeline()

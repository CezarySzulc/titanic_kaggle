from dowload_data import create_data_frame
from pandas import get_dummies, qcut, to_numeric
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re
import numpy as np
# # linear model
# from sklearn.linear_model import LogisticRegressionCV
# # xgb
# import xgboost as xgb


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

    def fillna_with_normal_distribution(self, df):
        # mask of NaN's
        mask = df.isnull()
        mu, sigma = df.mean(), df.std()
        df[mask] = np.random.normal(mu, sigma, size=mask.sum())
        return df

    def clear_data(self, data):
        # fill nan values in 'Age' and 'Fare' columns
        for _sex in data['Sex'].unique():
            mask_sex = data['Sex'] == _sex
            for _pclass in data['Pclass'].unique():
                mask = np.logical_and(
                    mask_sex,
                    data['Pclass'] == _pclass
                )
                data.loc[mask, 'Age'] = self.fillna_with_normal_distribution(
                    data.loc[mask, 'Age'])
                data.loc[mask, 'Fare'] = self.fillna_with_normal_distribution(
                    data.loc[mask, 'Fare'])

            data.loc[mask_sex, 'Age'] = self.fillna_with_normal_distribution(
                data.loc[mask_sex, 'Age']
            )
            data.loc[mask_sex, 'Fare'] = self.fillna_with_normal_distribution(
                data.loc[mask_sex, 'Fare']
            )
        data['Age'] = qcut(data.Age, q=5, labels=False)
        data['Fare'] = qcut(data.Fare, q=5, labels=False)
        # change 'Sex' columns into int type
        data = get_dummies(data, columns=['Sex'], drop_first=True)
        data = get_dummies(data, columns=['Embarked'], drop_first=True)
        # categorical 'Name' column
        data['Title'] = data['Name'].apply(lambda x: re.search('([A-Z][a-z]+)\.', x).group(1))
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
        data['Cabin_letter'] = data[data['Has_Cabin']]['Cabin'].apply(lambda x: re.search('([A-Z])+', x).group())
        data = get_dummies(data, columns=['Cabin_letter'])
        # drop unused values
        data.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

        print(data.info())
        print(data.describe())
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
        plt.figure()
        sns.distplot(self.train_data[self.train_data['Sex_male'] > 0]['Age'])
        plt.figure()
        sns.distplot(self.train_data[self.train_data['Sex_male'] == 0]['Age'])
        plt.show()

    def pipeline(self):
        columns = self.test_data.columns.tolist()
        columns.remove('PassengerId')
        # Randorm forest tree clf
        model = RandomForestClassifier(
            random_state=43,
            n_jobs=-1,
            n_estimators=50,
            # min_samples_split=7
        )
        param_grid = {
            # 'n_estimators': np.arange(5, 100, 5),
            'max_depth': np.arange(5, 15),
            'min_samples_split': np.arange(2, 10)
        }
        model = GridSearchCV(model, param_grid=param_grid, cv=5)
        model.fit(self.train_data[columns], self.train_label)
        print('Model best params: {}'.format(model.best_params_))
        print('Model best score: {}'.format(model.best_score_))
        print(model.best_estimator_)
        if self.need_valid_data:
            print(model.score(self.valid_data[columns], self.valid_label))
            print(classification_report(model.predict(self.valid_data[columns]), self.valid_label))
        self.test_data['Survived'] = model.predict(self.test_data[columns])

    def save_test_output(self, name):
        self.test_data[['PassengerId', 'Survived']].to_csv('data/{}.csv'.format(name), index=False)


my_model = Model()
# my_model.visualization()
my_model.pipeline()
my_model.save_test_output('random_forest')

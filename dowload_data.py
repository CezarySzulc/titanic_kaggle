from pandas import read_csv


def create_data_frame():
    # train data
    train_data = read_csv('data/train.csv')
    train_label = train_data['Survived']
    train_data.drop(axis=1, labels='Survived', inplace=True)
    # valid data

    # test data
    test_data = read_csv('data/test.csv')
    return train_data, train_label, test_data


if __name__ == '__main__':
    create_data_frame()

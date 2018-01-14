from pandas import read_csv
from sklearn.model_selection import train_test_split


def create_data_frame(valid_size=0.2):
    """
    create train, valid and test sets,
    default valid set is 20% of train data
    """
    # train data and valid data
    train_data = read_csv('data/train.csv')
    train_label = train_data['Survived']
    # train_data.drop(axis=1, labels='Survived', inplace=True)
    train_data, valid_data, train_label, valid_label = train_test_split(
        train_data, train_label, test_size=valid_size, random_state=43
    )
    # test data
    test_data = read_csv('data/test.csv')
    return train_data, train_label, valid_data, valid_label, test_data


if __name__ == '__main__':
    _train_data, _train_label, _valid_data, _valid_label, _test_data = create_data_frame()
    print(_train_data.shape)
    print(_valid_data.shape)

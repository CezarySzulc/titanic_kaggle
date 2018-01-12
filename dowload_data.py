import pandas as pd


def create_data_frame():
    train_data = pd.read_csv('data/train.csv')
    print(train_data.head())


if __name__ == '__main__':
    create_data_frame()

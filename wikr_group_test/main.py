import os
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from random import randint
import pickle

# path to train data
DATA_PATH = os.path.join(os.getcwd(), 'data', 'Houses_data.csv')
# saved weights for trained model
LOAD_MODEL = False
# save model weights of trained model
SAVE_MODEL = False
# list of available schools. Needed due to script features
SCHOOLS = ['edison', 'adams', 'parker', 'edge', 'harris', 'crest']

"""
There are 3 types of status, ['sld' 'act' 'pen']. And it is clear that 'sold' or 'pending' statuses seems to be more
reliable, and 'actual' statuses are overpriced. But our dataset is SUUUCH small, so I decide to leave them. If you
want to exclude samples with 'actual' status from training process, please change field below to True
"""
Exclude_actual = False


"""Data for testing model"""
# area in square feet
SIZE = 3214
# lot size category, specific feature, range: 0 - 11
LOT = 3
# number of bathrooms
BATH = 2
# number of bedrooms
BED = 3
# year of building
YEAR = 1980
# garage size
GARAGE = 1
# nearest elementary school, look at SCHOOLS above
SCHOOL = 'adams'


def get_learning_data():
    """Get train data

    :return: dataframe
    """
    df = pd.read_csv(DATA_PATH)
    df = df[df['Size'] < 1800]
    if Exclude_actual:
        df = df[df['Status'] != 'sld']
    return df


def input_to_df():
    """ Converting input data to df to make it later possible make a prediction

    :return: dataframe
    """
    parameters = {}
    parameters['Size'] = SIZE / 1000
    parameters['Lot'] = int(LOT)
    parameters['Bath'] = BATH * 1.0
    parameters['Bed'] = int(BED)
    parameters['Year'] = int(YEAR)
    parameters['Age'] = (1970 - parameters['Year']) / 10
    parameters['Garage'] = int(GARAGE)
    parameters['Status'] = 'act'
    parameters['elem'] = SCHOOL.lower()
    tmp = pd.DataFrame([parameters])
    tmp = tmp.ix[:, ['Price', 'Size', 'Lot', 'Bath', 'Bed', 'Year', 'Age', 'Garage', 'Status', 'elem']]
    return tmp


def prepare_data(df, data_type='test'):
    """ Leading all data to general view, both train or test dataframes

    :param df: dataframe
    :param data_type: dataframe
    :return: tuple of separated data
    """

    df = shuffle(df)

    if data_type == 'train':
        df = df.drop(labels=['id', 'Year', 'Status'], axis=1)
        df = pd.get_dummies(df)
    else:
        df = df.drop(labels=['Year', 'Status'], axis=1)
        lst = list(df['elem'].unique())
        df = pd.get_dummies(df)
        for item_ in [item for item in SCHOOLS if item not in lst]:
            df['elem_' + item_] = 0
        df['Price'] = 0

    Y = df['Price']
    df = df.drop(labels=['Price'], axis=1)
    X = df.ix[:, ['Size', 'Lot', 'Bath', 'Bed', 'Age', 'Garage', 'elem_adams', 'elem_crest', 'elem_edge',
                  'elem_edison', 'elem_harris', 'elem_parker']]
    return X, Y


def learn_model():
    """ xgboost showed best results on accuracy and coefficient of determination

    :return: model
    """
    df = get_learning_data()
    X, Y = prepare_data(df, data_type='train')
    print('Learning data size: ', len(X))
    model = xgb.XGBRegressor()
    model.fit(X, Y)
    if SAVE_MODEL:
        pickle.dump(model, open("model.dat", "wb"))
    return model


if __name__ == '__main__':

    if LOAD_MODEL:
        final_model = pickle.load(open("model.dat", "rb"))
    else:
        final_model = learn_model()

    print('='*30)
    X, Y = prepare_data(get_learning_data(), 'train')
    random_int = randint(0, len(Y))
    print('Predicting price for random sample from train data')
    print('Real value: ', list(Y[random_int:random_int+1])[0])
    print('Predicted value: ', list(final_model.predict(X[random_int:random_int+1]))[0])

    print()
    print('=' * 30)
    X, Y = prepare_data(input_to_df(), 'test')
    print('Predicting price for new data: ')
    print('Predicted value: ', list(final_model.predict(X[0:1]))[0])

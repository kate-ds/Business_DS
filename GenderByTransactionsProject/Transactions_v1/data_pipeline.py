import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from pathlib import Path
import warnings


from gensim.models import word2vec

DATA_ROOT = Path('./data/')
MODELS_PATH = Path('./models/')

# input
TRANSACTIONS_DATASET_PATH = DATA_ROOT / 'transactions.csv'
GENDER_TRAIN_DATASET_PATH = DATA_ROOT / 'gender_train.csv'
MCC_DATASET_PATH = DATA_ROOT / 'tr_mcc_codes.csv'
TYPES_DATASET_PATH = DATA_ROOT / 'tr_types.csv'
Y_TEST = DATA_ROOT / 'gender_test_kaggle_sample_submission.csv'
STOP_WORDS = DATA_ROOT / 'stopwords.txt'

# output
PREPARED_TRAIN_DATASET_PATH = DATA_ROOT / 'train_prepared.csv'
PREPARED_TEST_DATASET_PATH = DATA_ROOT / 'test_prepared.csv'

MODEL_FILE_PATH = MODELS_PATH / 'model.pkl'

# --------------------- data -------------------------------------

transactions = pd.read_csv(TRANSACTIONS_DATASET_PATH)
gender = pd.read_csv(GENDER_TRAIN_DATASET_PATH)
test_id = pd.read_csv(Y_TEST)

df = pd.merge(transactions, gender, on='customer_id', how='left')

y_train = gender.gender
print('y: ', y_train.shape)
X_train = df.loc[df['customer_id'].isin(gender.customer_id.values)].drop(columns='gender')
X_test = df.loc[df['customer_id'].isin(test_id.customer_id.values)].drop(columns='gender')

warnings.filterwarnings('ignore')

disbalance = y_train.value_counts()[0] / y_train.value_counts()[1]
# ----------------------- pipeline ---------------------------------

def time_preprocessing(column, log=True):
    '''
    Time converting to seconds, outliers preprocessing, hour column
    ----------------------------------
    parameters:
    column: pandas Series, time column
    log: bool, printing information about outliers

    returns:
    new_times: numpy array, time in datetime
    time_array: numpy array, time of transactions in seconds
    hours_array: numpy array, hour of transactions

    '''
    times = [time.split(':') for time in column]
    hour_err = 0
    minutes_err = 0
    second_err = 0
    time_in_seconds = []
    hours = []

    for time in times:
        if int(time[0]) > 23:
            hour_err += 1
            time[0] = 23
        if int(time[1]) > 59:
            minutes_err += 1
            time[1] = 59
        if int(time[2]) > 59:
            second_err += 1
            time[2] = 59
        hours.append(int(time[0]))
        time = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
        time_in_seconds.append(time)

    time_array = np.array(time_in_seconds)
    hours_array = np.array(hours)

    if log:
        print(f'Всего ошибок:\nчасы:{hour_err}\nминуты: {minutes_err}\nсекунды: {second_err}')

    return time_array, hours_array


def spent_for_a_month(df):
    """всего потрачено за неделю"""
    gp = df.groupby(['customer_id', 'month'])['spent'].sum().rename('spent_for_a_month', inplace=True)
    df = pd.merge(df, gp, on=['customer_id', 'month'], how="left")
    return df

def spent_for_a_week(df):
    """всего потрачено за неделю"""
    gp = df.groupby(['customer_id', 'week'])['spent'].sum().rename('spent_for_a_week', inplace=True)
    df = pd.merge(df, gp, on=['customer_id', 'week'], how="left")
    return df

def income_for_a_month(df):
    """всего получено за месяц"""
    gp = df.groupby(['customer_id', 'month'])['income'].sum().rename('income_for_a_month', inplace=True)
    df = pd.merge(df, gp, on=['customer_id', 'month'], how="left")
    return df


# средняя цена за покупку у пользователя
def median_purchase(df):
    gp = df.groupby('customer_id')['spent'].median().rename('median_purchase', inplace=True)
    df = pd.merge(df, gp, on=['customer_id'], how="left")
    return df


# если цена превышает пользовательскую среднюю, то будем считать покупку дорогой
def expensive(df):
    df['more_than_median'] = df['amount'] - df['median_purchase']
    df['expensive'] = np.where(df['more_than_median'].values > 0, 1, 0)
    return df


# посчитаем количество дорогих покупок для пользователя
def number_of_expensive(df):
    gp = df.groupby('customer_id')['expensive'].sum().rename('number_of_expensive', inplace=True)
    df = pd.merge(df, gp, on=['customer_id'], how="left")
    return df


class TfidfVect(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # self.X = X
        self.columns = []
        self.cols_for_merge = ['diff_income_spent', 'dept', 'min_spent_for_a_month',
                               'max_spent_for_a_month', 'mean_spent_for_a_month',
                               'min_spent_for_a_week', 'max_spent_for_a_week', 'mean_spent_for_a_week',
                               'min_income_for_a_month', 'max_income_for_a_month',
                               'mean_income_for_a_month', 'more_than_mean', 'median_purchase',
                               'more_than_median', 'expensive', 'number_of_expensive']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # колонка количества дней
        X['n_day'] = np.array([int(el.split(' ')[0]) for el in X['tr_datetime'].values])
        # колонка месяца (возьмем усредненно 30 дней)
        X['month'] = X['n_day'].values // 30
        # колонка недели
        X['week'] = X['n_day'].values // 7
        # колонка времени
        X['time'] = np.array([el.split(' ')[1] for el in X['tr_datetime'].values])
        # обработка временной колонки
        X['time'], X['hour'] = time_preprocessing(X['time'], log=False)
        # колонка затрат
        X['spent'] = np.where(X['amount'].values < 0, abs(X['amount'].values), 0)
        # колонка пополнений
        X['income'] = np.where(X['amount'].values > 0, X['amount'].values, 0)
        self.columns = X.columns
        # всего потрачено за месяц
        X = spent_for_a_month(X)
        # всего потрачено за неделю
        X = spent_for_a_week(X)
        # всего получено за месяц
        X = income_for_a_month(X)
        # разница полученного и потраченного (может быть долг или накопления)
        X['diff_income_spent'] = X['income'] - X['spent']
        # есть ли задолженность
        X['dept'] = np.where(X['diff_income_spent'] < 0, 1, 0)

        gp = X.groupby('customer_id').agg(min_spent_for_a_month=('spent_for_a_month', 'min'),
                                          max_spent_for_a_month=('spent_for_a_month', 'max'),
                                          mean_spent_for_a_month=('spent_for_a_month', 'mean'),
                                          min_spent_for_a_week=('spent_for_a_week', 'min'),
                                          max_spent_for_a_week=('spent_for_a_week', 'max'),
                                          mean_spent_for_a_week=('spent_for_a_week', 'mean'),
                                          min_income_for_a_month=('income_for_a_month', 'min'),
                                          max_income_for_a_month=('income_for_a_month', 'max'),
                                          mean_income_for_a_month=('income_for_a_month', 'mean'))

        X = pd.merge(X, gp, on='customer_id', how='left')

        X['more_than_mean'] = np.where(X['mean_spent_for_a_month'] > 1e9, 1, 0)
        # средняя цена за покупку у пользователя
        X = median_purchase(X)
        # если цена превышает пользовательскую среднюю, то будем считать покупку дорогой
        X = expensive(X)
        # посчитаем количество дорогих покупок для пользователя
        X = number_of_expensive(X)
        # сбрасываем невостребованные колонки
        X.drop(columns=['term_id', 'tr_datetime'], inplace=True)

        # обрабатываем мсс-коды
        users = X.groupby('customer_id')['mcc_code'].apply(list).reset_index()
        model = word2vec.Word2Vec(users['mcc_code'], vector_size=80, window=3, workers=4)
        w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))
        idf = TfidfVect(w2v)
        idf.fit(users['mcc_code'])
        data = idf.transform(users['mcc_code'])
        matrix = pd.DataFrame(data, columns=['mcc_{}'.format(i) for i in range(data.shape[1])])
        users = pd.concat((users, matrix), axis=1).drop(columns=['mcc_code'])

        # обрабатываем tr_type
        types = X.groupby("customer_id")['tr_type'].apply(list).reset_index()
        model2 = word2vec.Word2Vec(types['tr_type'], vector_size=20, window=3, workers=4)
        w2v2 = dict(zip(model2.wv.index_to_key, model2.wv.vectors))
        idf2 = TfidfVect(w2v2)
        idf2.fit(types['tr_type'])
        data2 = idf2.transform(types['tr_type'])
        matrix2 = pd.DataFrame(data2, columns=['type_{}'.format(i) for i in range(data2.shape[1])])
        users = pd.concat((users, matrix2), axis=1)

        # объединение датафреймов
        X = X.groupby('customer_id')[self.cols_for_merge].mean()
        X = pd.merge(X, users, on='customer_id', how='left')
        print('X: ', X.shape)

        return X


pip = Pipeline([('data_preprocessor', DataPreprocessor()),
                ('catboost', CatBoostClassifier(learning_rate=0.08,
                                                n_estimators=350,
                                                depth=4,
                                                class_weights=[1, disbalance],
                                                custom_metric='AUC',
                                                silent=True))])

valu = pip.fit(X_train, y_train).predict_proba(X_train)
print(1)
print(valu[:30])

cv_scores = cross_val_score(pip, X_train, y_train, cv=16, scoring='roc_auc')
print(cv_scores)



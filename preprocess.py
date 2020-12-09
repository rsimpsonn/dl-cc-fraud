import csv
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

'''def get_csv_data(filepath):
    fields = [] 
    rows = [] 
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader) 
        for row in csvreader: 
            rows.append(row)
    print("Total no. of rows: %d"%(csvreader.line_num))
    print('Field names are:' + ', '.join(field for field in fields))
    print('\nFirst 5 rows are:\n') 
    for row in rows[:5]: 
        # parsing each column of a row 
        for col in row: 
            print("%10s"%col), 
        print('\n') 
    return np.array(fields), np.array(rows, dtype=np.float32)'''

def preprocess_cred_crd(filepath):
    df = pd.read_csv("data/creditcard.csv")
    data = df.drop(['Time'], axis=1)

    print(data['Amount'])
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    print(data['Amount'])
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis=1)

    y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis=1)

    X_train = X_train.values
    X_test = X_test.values

    return X_train, X_test, y_test

def preprocess_cred_crd_seq(filepath):

    trans = pd.read_csv(filepath, na_filter=True)
    labels = trans['Class']
    trans = trans.drop(['Class'], axis=1)
    trans = trans.drop(['Time'], axis=1)

    rolling_window_size = 40

    windows = np.array([np.array(trans[i:i + rolling_window_size]) for i in range(len(trans) - rolling_window_size)])
    window_labels = labels[rolling_window_size:]

    offset = int(len(windows) * 0.7)

    return windows[:offset], window_labels[:offset], windows[offset:], window_labels[offset:]

def gen_normalized_sim_data(inf, out):
    df = pd.read_csv(inf)
    data = df.drop(df.columns[0], axis=1)
    data = data.assign(ind=(df['first'] + '_' + df['last']).astype('category').cat.codes)
    data = data.assign(mer=(df['merchant']).astype('category').cat.codes)
    data = data.assign(cc=(df['cc_num']).astype('category').cat.codes)
    data = data.assign(cat=(df['category']).astype('category').cat.codes)
    data = data.assign(gen=(df['gender']).astype('category').cat.codes)
    data = data.assign(str=(df['street']).astype('category').cat.codes)
    data = data.assign(ct=(df['city']).astype('category').cat.codes)
    data = data.assign(st=(df['state']).astype('category').cat.codes)
    data = data.assign(zcode=(df['zip']).astype('category').cat.codes)
    data = data.assign(jobtype=(df['job']).astype('category').cat.codes)
    data = data.assign(birth=(df['dob']).astype('category').cat.codes)

    data['time'] = pd.to_datetime(data['trans_date_trans_time'])
    first_trans = data.iloc[0]
    data['time'] = data['time'].apply(lambda x: (x - first_trans['time']).total_seconds())

    data = data.drop(['first', 'last', 'unix_time', 'trans_date_trans_time', 'category', 'gender', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num', 'merchant', 'cc_num'], axis=1)

    min_max_scaler = MinMaxScaler()

    data['ind'] = min_max_scaler.fit_transform(data[['ind']].values)
    data['long'] = min_max_scaler.fit_transform(data[['long']].values)
    data['lat'] = min_max_scaler.fit_transform(data[['lat']].values)
    data['cc'] = min_max_scaler.fit_transform(data[['cc']].values)
    data['mer'] = min_max_scaler.fit_transform(data[['mer']].values)
    data['cat'] = min_max_scaler.fit_transform(data[['cat']].values)
    data['gen'] = min_max_scaler.fit_transform(data[['gen']].values)
    data['str'] = min_max_scaler.fit_transform(data[['str']].values)
    data['ct'] = min_max_scaler.fit_transform(data[['ct']].values)
    data['st'] = min_max_scaler.fit_transform(data[['st']].values)
    data['zcode'] = min_max_scaler.fit_transform(data[['zcode']].values)
    data['jobtype'] = min_max_scaler.fit_transform(data[['jobtype']].values)
    data['birth'] = min_max_scaler.fit_transform(data[['birth']].values)
    data['merch_lat'] = min_max_scaler.fit_transform(data[['merch_lat']].values)
    data['merch_long'] = min_max_scaler.fit_transform(data[['merch_long']].values)
    data['city_pop'] = min_max_scaler.fit_transform(data[['city_pop']].values)

    data.to_csv(out, index=False)

def preprocess_cred_crd_sim(train, test):
    if not(os.path.isfile("data/fraudTrainNormalized.csv")):
        gen_normalized_sim_data(train, "data/fraudTrainNormalized.csv")

    if not(os.path.isfile("data/fraudTestNormalized.csv")):
        gen_normalized_sim_data(test, "data/fraudTestNormalized.csv")

    train_data = pd.read_csv("data/fraudTrainNormalized.csv")
    train_data = train_data.drop(train_data.columns[0], axis=1)

    train_data = train_data.drop(['time'], axis=1)

    train_data['amt'] = StandardScaler().fit_transform(train_data['amt'].values.reshape(-1, 1))
    train_data = train_data.drop(['is_fraud'], axis=1)

    test_data = pd.read_csv("data/fraudTestNormalized.csv")
    test_data = test_data.drop(test_data.columns[0], axis=1)

    test_data = test_data.drop(['time'], axis=1)
    test_labels = test_data['is_fraud']

    test_data['amt'] = StandardScaler().fit_transform(test_data['amt'].values.reshape(-1, 1))
    test_data = test_data.drop(['is_fraud'], axis=1)

    #print(train_data.columns)

    #print(test_data.head())

    #print(train_data.values.shape)

    return train_data.values, test_data.values, test_labels.values

def preprocess_normalized_sim_lstm(filepath):
    df = pd.read_csv(filepath, na_filter=True)
    df = df.sort_values(by=["ind"])
    rolling_window_size = 40
    pre = []
    windows = []
    for index, row in df.iterrows():
        if len(pre) == 0 or pre[-1]['ind'] == row['ind']:
            pre.append(row)
        if (len(pre) == rolling_window_size):
            b = np.array([x.values for x in pre])
            # print(b.shape)
            windows.append(b)
            pre = []
        if len(pre) != 0 and pre[-1]['ind'] != row['ind']:
            pre = []
            pre.append(row)
    print(np.array(windows).shape)
    return np.array(windows)

    # windows = np.array([np.array(df[i:i + rolling_window_size]) for i in range(len(df) - rolling_window_size)])
    # window_labels = labels[rolling_window_size:]
    # offset = int(len(windows) * 0.7)
    # split = windows[:offset], window_labels[:offset], windows[offset:], window_labels[offset:]
    # return split


#preprocess_normalized_sim_lstm("data/new.csv")


def preprocess_together(filepath, trainpath, testpath):
    _, data = get_csv_data(filepath)
    _, traindata = get_csv_data(trainpath)
    _, testdata = get_csv_data(testpath)
    pass

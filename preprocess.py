import csv
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
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

    rolling_window_size = 10

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
    data = data.assign(city=(df['city']).astype('category').cat.codes)
    data = data.assign(st=(df['state']).astype('category').cat.codes)
    data = data.assign(zcode=(df['zip']).astype('category').cat.codes)
    data = data.assign(jobtype=(df['job']).astype('category').cat.codes)
    data = data.assign(birth=(df['dob']).astype('category').cat.codes)

    data['time'] = pd.to_datetime(data['trans_date_trans_time'])
    first_trans = data.iloc[0]
    data['time'] = data['time'].apply(lambda x: (x - first_trans['time']).total_seconds())

    data = data.drop(['first', 'last', 'unix_time', 'trans_date_trans_time', 'category', 'gender', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num', 'merchant', 'cc_num'], axis=1)

    data.to_csv(out)

def preprocess_cred_crd_sim():
    if not(os.path.isfile("data/fraudTrainNormalized.csv")):
        gen_normalized_sim_data("data/fraudTrain.csv", "data/fraudTrainNormalized.csv")

    if not(os.path.isfile("data/fraudTestNormalized.csv")):
        gen_normalized_sim_data("data/fraudTest.csv", "data/fraudTestNormalized.csv")




def preprocess_together(filepath, trainpath, testpath):
    _, data = get_csv_data(filepath)
    _, traindata = get_csv_data(trainpath)
    _, testdata = get_csv_data(testpath)
    pass


gen_normalized_sim_data()
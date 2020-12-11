import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


# --------------------------Filepaths---------------


eu_filepath = "data/creditcard.csv"
sim_train_path = "data/fraudTrain.csv"
sim_test_path = "data/fraudTest.csv"
sim_norm_train_path = "data/fraudTrainNormalized.csv"
sim_norm_test_path = "data/fraudTestNormalized.csv"


# ------------------EU Data AutoEncoder Preprocess-----------


def preprocess_eu_ae():
    df = pd.read_csv(eu_filepath)
    df = df.drop(['Time'], axis=1)
    df = normalize(df, "Amount", True)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train = train[train.Class == 0]
    train = train.drop(['Class'], axis=1)
    labels = test['Class']
    test = test.drop(['Class'], axis=1)
    return train.values, test.values, labels


# -------------------Simulated Data CSV Transform---------------


def gen_sim_csv(inf, out):
    df = pd.read_csv(inf)
    df = df.assign(ind=(df['first'] + '_' + df['last']).astype('category').cat.codes)
    df = df.assign(mer=(df['merchant']).astype('category').cat.codes)
    df = df.assign(cc=(df['cc_num']).astype('category').cat.codes)
    df = df.assign(cat=(df['category']).astype('category').cat.codes)
    df = df.assign(gen=(df['gender']).astype('category').cat.codes)
    df = df.assign(str=(df['street']).astype('category').cat.codes)
    df = df.assign(ct=(df['city']).astype('category').cat.codes)
    df = df.assign(st=(df['state']).astype('category').cat.codes)
    df = df.assign(zcode=(df['zip']).astype('category').cat.codes)
    df = df.assign(jobtype=(df['job']).astype('category').cat.codes)
    df = df.assign(birth=(df['dob']).astype('category').cat.codes)

    df['time'] = pd.to_datetime(df['trans_date_trans_time'])
    first_trans = df.iloc[0]
    df['time'] = df['time'].apply(lambda x: (x - first_trans['time']).total_seconds())

    df = df.drop(['first', 'last', 'unix_time', 'trans_date_trans_time', 'category', 'gender', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num', 'merchant', 'cc_num'], axis=1)

    columns_std = [("ind", False), ("long", True), ("lat", True), ("cc", False),
                   ("mer", False), ("cat", False), ("gen", False), ("str", False),
                   ("ct", False), ("st", False), ("zcode", False), ("jobtype", False),
                   ("birth", False), ("merch_lat", True), ("merch_long", True),
                   ("city_pop", True)]
    for col, standard in columns_std:
        df = normalize(df, col, standard)
    df.to_csv(out, index=False)


# ------------------Simulated Data AutoEncoder Preprocess---------------


def preprocess_sim_ae():
    if not(os.path.isfile(sim_norm_train_path)):
        gen_sim_csv(sim_test_path, sim_norm_train_path)
    if not(os.path.isfile(sim_norm_test_path)):
        gen_sim_csv(sim_test_path, sim_norm_test_path)
    train = pd.read_csv(sim_norm_train_path)
    test = pd.read_csv(sim_norm_test_path)
    train = train.drop(['time'], axis=1)
    train = normalize(train, "amt", True)
    train = train[train.is_fraud == 0]
    train = train.drop(['is_fraud'], axis=1)
    test = test.drop(['time'], axis=1)
    labels = test['is_fraud']
    test = normalize(test, "amt", True)
    test = test.drop(['is_fraud'], axis=1)
    return train.values, test.values, labels.values


# ---------------Simulated Data LSTM Preprocess-----------------


def preprocess_sim_lstm():
    if (os.path.isfile("data/lstm_data")):
        pii = pickle.open("data/lstm_data")
        a, b, c, d = pickle.load(pii)
        pii.close()
        return a, b, c, d
    df = pd.read_csv(sim_norm_train_path, na_filter=True)
    df = df.sort_values(by=["ind"])
    rolling_window_size = 40
    pre = []
    windows = []
    labels = []
    x = []
    for idx, row in df.iterrows():
        # row.pop("is_fraud")
        # print("x shape: " + str(np.array(x).shape))
        if len(x) == 0 or x[-1]['ind'] == row['ind']:
            x.append(row)
        if len(x) != 0 and x[-1]['ind'] != row['ind']:
            # print("true")
            pre.append(x)
            x = []
            x.append(row)
    for same_usr in pre:
        if len(same_usr) >= rolling_window_size:
            usr_windows = [same_usr[i:i+rolling_window_size] for i in range(len(same_usr) - rolling_window_size)]
            # print(usr_windows)
            usr_labels = [same_usr[i+rolling_window_size] for i in range(len(same_usr) - rolling_window_size)]
            for usr_window, usr_label in zip(usr_windows, usr_labels):
                # print(usr_window)
                # print("DONE")
                # print(usr_label)
                # print("AFE")
                xtay = []
                for d in usr_window:
                    w = d.copy()
                    w.pop("is_fraud")
                    xtay.append(w)
                windows.append(xtay)
                # windows.append([d.pop("is_fraud") for d in usr_window])
                labels.append(usr_label["is_fraud"])
    windows = np.array(windows)
    labels = np.array(labels)
    print("finished")
    print(windows.shape)
    print(labels.shape)
    offset = int(len(windows) * 0.7)
    st = pickle.open("data/lstm_data")
    st.dump(windows[:offset], labels[:offset], windows[offset:], labels[offset:])
    st.close()
    return windows[:offset], labels[:offset], windows[offset:], labels[offset:]
    
    # for index, row in df.iterrows():
    #     if len(pre) == 0 or pre[-1]['ind'] == row['ind']:
    #         pre.append(row)
    #     if (len(pre) == rolling_window_size + 1):
    #         b = np.array([x.values for x in pre])
    #         windows.append(b)
    #         pre = []
    #     if len(pre) != 0 and pre[-1]['ind'] != row['ind']:
    #         pre = []
    #         pre.append(row)
    # return np.array(windows)


# ---------------------Helpers-------------


def normalize(df, column, standard=False):
    if standard:
        df[column] = MinMaxScaler().fit_transform(df[[column]].values)
    else:
        df[column] = StandardScaler().fit_transform(df[column].values.reshape(-1, 1))
    return df
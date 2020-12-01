import csv
import numpy as np
import pandas as pd

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
    trans = pd.read_csv(filepath)
    trans = trans.drop(['Time'], axis=1)
    offset = int(len(trans) * 0.7)
    traindata = trans.loc[:offset, :]
    traindata = traindata[traindata['Class'] == 0]
    traindata = traindata.drop(['Class'], axis=1)
    testdata = trans.loc[offset:, :]

    testlabels = testdata['Class']
    testdata = testdata.drop(['Class'], axis=1)

    return traindata.to_numpy(dtype=np.float32), testdata.to_numpy(dtype=np.float32), testlabels.to_numpy(dtype=np.float32)

def preprocess_sim_cred_crd(trainpath, testpath):
    _, traindata = get_csv_data(trainpath)
    _, testdata = get_csv_data(testpath)
    print(traindata)
    return traindata, testdata

def preprocess_together(filepath, trainpath, testpath):
    _, data = get_csv_data(filepath)
    _, traindata = get_csv_data(trainpath)
    _, testdata = get_csv_data(testpath)
    pass


preprocess_cred_crd("data/creditcard.csv")
import csv

def get_csv_data(filepath):
    with open(filepath, 'r') as csvfile
        csvreader = csv.reader(csvfile)
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
    return fields, rows

def preprocess_cred_crd(filepath):
    _, data = get_csv_data(filepath)
    offset = int(len(data) * 0.7)
    traindata = data[:offset]
    testdata = data[offset:]
    return traindata, testdata

def preprocess_sim_cred_crd(trainpath, testpath):
    _, traindata = get_csv_data(trainpath)
    _, testdata = get_csv_data(testpath)
    return traindata, testdata

def preprocess_together(filepath, trainpath, testpath):
    _, data = get_csv_data(filepath)
    _, traindata = get_csv_data(trainpath)
    _, testdata = get_csv_data(testpath)
    pass
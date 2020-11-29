
cred_crd_filepath = "data/creditcard.csv"
sim_cred_crd_trainpath = "data/fraudTrain.csv"
sim_cred_crd_testpath = "data/fraudTest.csv"

def run_w_cred_crd():
    traindata, testdata = preprocess_cred_crd(cred_crd_filepath)
    seq_network(traindata, testdata)

def run_w_sim_cred_crd():
    traindata, testdata = preprocess_sim_cred_crd(sim_cred_crd_trainpath,
                                                  sim_cred_crd_testpath)
    seq_network(traindata, testdata)

def run_w_both_data():
    traindata, testdata = preprocess_together(cred_crd_filepath,
                                              sim_cred_crd_trainpath,
                                              sim_cred_crd_testpath)
    seq_network(traindata, testdata)
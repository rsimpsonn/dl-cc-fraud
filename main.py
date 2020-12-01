import sys
from preprocess import preprocess_cred_crd, preprocess_sim_cred_crd 
from network import autoencoder_network, lstm_network, transformer_network

cred_crd_filepath = "data/creditcard.csv"
sim_cred_crd_trainpath = "data/fraudTrain.csv"
sim_cred_crd_testpath = "data/fraudTest.csv"

def main():
	network_type = sys.argv[1]
	data_type = sys.argv[2]

	if data_type == "EU":
		traindata, testdata, testlabels = preprocess_cred_crd(cred_crd_filepath)
	elif data_type == "SIM":
		traindata, testdata = preprocess_sim_cred_crd(sim_cred_crd_trainpath, sim_cred_crd_testpath)
	elif data_type == "BOTH":
		pass
	else:
		print("Please put a valid data type (i.e. $ python3 main.py NETWORK_TYPE DATA_TYPE, where DATA_TYPE is either EU or SIM)")
		return

	if network_type == "A":
		autoencoder_network(traindata, testdata, testlabels)
	elif network_type == "L":
		lstm_network(traindata, testdata)
	elif network_type == "T":
		transformer_network(traindata, testdata)
	else:
		print("Please put a valid network type (i.e. $ python3 main.py NETWORK_TYPE DATA_TYPE, where DATA_TYPE is either A, L or T)")

if __name__ == '__main__':
    main()
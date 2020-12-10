import sys
from preprocess import preprocess_eu_ae, preprocess_sim_ae, preprocess_sim_lstm
from network import autoencoder_network, lstm_network

def main():
	network_type = sys.argv[1]
	data_type = sys.argv[2]
	if network_type == "AE" and data_type == "EU":
		train, test, labels = preprocess_eu_ae()
		autoencoder_network(train, test, labels)
	elif network_type == "AE" and data_type == "SIM":
		train, test, labels = preprocess_sim_ae()
		autoencoder_network(train, test, labels)
	elif network_type == "LSTM" and data_type == "SIM":
		train_d, train_l, test_d, test_l= preprocess_sim_lstm()
		lstm_network(train_d, train_l, test_d, test_l)
	else:
		cmd_line_err_message = "Please put a valid network type (i.e. $ python3 main.py NETWORK_TYPE DATA_TYPE, where NETWORK_TYPE is either AE or LSTM and DATA_TYPE is either EU or SIM"
		print(cmd_line_err_message)

if __name__ == '__main__':
    main()
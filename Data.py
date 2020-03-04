from sklearn import datasets
import csv
import numpy as np
from sklearn.datasets.base import Bunch

def load_my_dataset():
    with open('Code_Data.csv') as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = 7020 #number of data rows, don't count header
        n_features = 10 #number of columns for features, don't count target column
        feature_names = ['Area','Aspect_ratio','Solidity','m1','m2','m3','m4','m5','m6','m7'] #adjust accordingly
        target_names = ["Ain","Alif","Aray","Bari_yeh","Bay",
            			"Chasmi_hay","Chey","Chotti_yeh","Daal_with_hard_d","Daal_with_soft_d",
            			"Duaad","Dzay","Fay","Gaeen","Gaff","gool_hay","hamza","hay","jeem",
            			"kaaf","kaf","khay","laam","meem","noon","n_gunah","paay","ray",
            			"say","seen","sheen","suaad","tay","tey","tuey","wow","zaal","zay","zuey"] #adjust accordingly
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            target[i] = np.asarray(sample[-1], dtype=np.int)

    return Bunch(data=data, target=target, feature_names=feature_names, target_names=target_names)

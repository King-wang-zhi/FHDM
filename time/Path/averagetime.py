import numpy as np
import matplotlib.pyplot as plt
data1_accurate = np.loadtxt("client_2nn_FL.txt")
data2_accurate = np.loadtxt("client_2nn_LDP.txt")
data3_accurate = np.loadtxt("client_2nn_both.txt")
data4_accurate = np.loadtxt("client_cnn_FL.txt")
data5_accurate = np.loadtxt("client_cnn_LDP.txt")
data6_accurate = np.loadtxt("client_cnn_both.txt")

x1 = data1_accurate[:, 1]
x2 = data2_accurate[:, 1]
x3 = data3_accurate[:, 1]
x4 = data4_accurate[:, 1]
x5 = data5_accurate[:, 1]
x6 = data6_accurate[:, 1]

mean = np.mean(x1)
print("Mean:", mean)
mean = np.mean(x2)
print("Mean:", mean)
mean = np.mean(x3)
print("Mean:", mean)
mean = np.mean(x4)
print("Mean:", mean)
mean = np.mean(x5)
print("Mean:", mean)
mean = np.mean(x6)
print("Mean:", mean)
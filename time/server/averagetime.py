import numpy as np
import matplotlib.pyplot as plt
data1_accurate = np.loadtxt("Pneumonia/server_2nn_FL.txt")
data2_accurate = np.loadtxt("Blood/server_2nn_FL.txt")
data3_accurate = np.loadtxt("Path/server_2nn_FL.txt")
data4_accurate = np.loadtxt("Pneumonia/server_cnn_FL.txt")
data5_accurate = np.loadtxt("Blood/server_cnn_FL.txt")
data6_accurate = np.loadtxt("Path/server_cnn_FL.txt")

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

print("###################################")
data1_accurate = np.loadtxt("Pneumonia/server_2nn_LDP.txt")
data2_accurate = np.loadtxt("Blood/server_2nn_LDP.txt")
data3_accurate = np.loadtxt("Path/server_2nn_LDP.txt")
data4_accurate = np.loadtxt("Pneumonia/server_cnn_LDP.txt")
data5_accurate = np.loadtxt("Blood/server_cnn_LDP.txt")
data6_accurate = np.loadtxt("Path/server_cnn_LDP.txt")

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

print("###################################")
data1_accurate = np.loadtxt("Pneumonia/server_2nn_both.txt")
data2_accurate = np.loadtxt("Blood/server_2nn_both.txt")
data3_accurate = np.loadtxt("Path/server_2nn_both.txt")
data4_accurate = np.loadtxt("Pneumonia/server_cnn_both.txt")
data5_accurate = np.loadtxt("Blood/server_cnn_both.txt")
data6_accurate = np.loadtxt("Path/server_cnn_both.txt")

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
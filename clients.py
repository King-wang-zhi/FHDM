import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
from getDataPneumoniamnist import GetDataSetpneumoniamnist
from getDataBloodmnist import GetDataSetbloodmnist
from getDataPathmnist import GetDataSetpathmnist
from dp_mechanism import cal_sensitivity, Laplace, Gaussian_Simple, Gaussian_moment
import random
import time
from experience_mnist import *
from experience_Bloodmnist import *
from experience_Pneumoniamnist import *
from experience_Pathmnist import *
from utils import config
import shutil
import datetime
import time
import os.path
import sys
import torch.nn.functional as F

config = config()


class client(object):
    def __init__(self, trainDataSet, dev, dp_mechanism='no_dp', attack='no_attack', dp_epsilon=0.5, dp_delta=1e-5, dp_clip=1e-2, idxs=None):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

        self.dp_mechanism = 'no_dp'
        self.dp_epsilon = dp_epsilon  # 隐私预算
        self.dp_delta = dp_delta  # 松弛差分
        self.dp_clip = dp_clip  # 梯度裁剪
        self.idxs = idxs
        self.rate = random.random()
        # self.rate = 0.7
        self.turnlabel = 0
        self.attack = 'no_attack'
        self.rate = 0.5

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, dp_mechanism, num_comm,
                    now_num_comm, clientnum, attack, model_name):


        now = str(datetime.datetime.now())[:19]
        now = now.replace(":", "_")
        now = now.replace("-", "_")
        now = now.replace(" ", "_")

        src_dir = config.path.data_path
        path = config.path.result_path + str(model_name) + "_" + str(config.statistics.type) + "_" + str(
            now_num_comm)
        if os.path.exists(path) == False:
            os.mkdir(path)
        dst_dir = path + "/config.yaml"
        shutil.copy(src_dir, dst_dir)
        self.attack = attack
        Net.load_state_dict(global_parameters, strict=True)
        data_train_target = self.train_ds
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True, drop_last=True)
        # testTensorDataset = myClients.TestTensorDataset
        # testDataLoader = DataLoader(testTensorDataset, batch_size=localBatchSize, shuffle=True, drop_last=True)
        # dataloaders_target = {"train": self.train_dl, "val": testDataLoader}
        Loss_list = []
        # 进行预训练求DP比率
        start = time.time()
        for epoch in range(localEpoch - 1):
            running_loss = 0
            for data, label in self.train_dl:
                # data = data.reshape(10, 1, 28, 28)
                # print(data.size())
                data, label = data.to(self.dev), label.to(self.dev)

                # preds = Net(data)
                # loss = lossFun(preds, label)

                preds = F.log_softmax(Net(data), dim=1)
                loss = F.nll_loss(preds, label)

                loss.backward()

                opti.step()
                opti.zero_grad()
                running_loss += loss.item()
                # if epoch == localEpoch - 1:
                #     for out in outputs.cpu().detach().numpy():
                #         X.append(out)
                #         Y.append(1)
                #     for cla in label.cpu().detach().numpy():
                #         C.append(cla)
            with open('client_loss.txt', 'a') as f:
                f.write('%d %.3f\n' % (now_num_comm, running_loss))
            Loss_list.append(running_loss)

        proportion = (Loss_list[1] - 0) / (Loss_list[0] - 0)

        proportion = proportion ** 0.5

        threshold = 0.9
        w = (1 - (now_num_comm / num_comm) * threshold)

        self.rate = (1 - proportion) * w

        if self.rate < 0:
            self.rate = -self.rate

        # torch.save(Net.state_dict(), 'model.pth')

        for epoch in range(1):
            running_loss = 0

            X = []
            Y = []
            C = []
            Z = 0
            for data, label in self.train_dl:
                # data = data.reshape(10, 1, 28, 28)
                data, label = data.to(self.dev), label.to(self.dev)
                if Z == 0:
                    iDLG_data, iDLG_label = data, label
                    Z = 1
                preds = Net(data)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                # dy_dx = torch.autograd.grad(loss, Net.parameters())
                # if Z == 1:
                #     original_gradient = dy_dx
                #     Z = 2
                if dp_mechanism != 'no_dp':
                    self.clip_gradients(Net, dp_mechanism)
                opti.step()

                running_loss += loss.item()
                # 我们测时间不执行攻击
                # for out in preds.cpu().detach().numpy():
                #     X.append(out)
                #     Y.append(1)
                # for cla in label.cpu().detach().numpy():
                #     C.append(cla)
            with open('client_loss.txt', 'a') as f:
                f.write('%d %.3f\n' % (now_num_comm, running_loss))
            Loss_list.append(running_loss)

        if dp_mechanism != 'no_dp':
            if dp_mechanism != 'CDP':
                self.add_noise(Net, dp_mechanism)
        end = time.time()
        times = end - start
        '''
        # 测时间
        if dp_mechanism == 'Gaussian':
            if model_name == 'pneumoniamnist_2nn':
                with open('time/Pneumonia/client_2nn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pneumoniamnist_cnn':
                with open('time/Pneumonia/client_cnn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'bloodmnist_2nn':
                with open('time/Blood/client_2nn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'bloodmnist_cnn':
                with open('time/Blood/client_cnn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pathmnist_2nn':
                with open('time/Path/client_2nn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pathmnist_cnn':
                with open('time/Path/client_cnn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
        elif dp_mechanism == 'no_dp':
            if model_name == 'pneumoniamnist_2nn':
                with open('time/Pneumonia/client_2nn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pneumoniamnist_cnn':
                with open('time/Pneumonia/client_cnn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'bloodmnist_2nn':
                with open('time/Blood/client_2nn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'bloodmnist_cnn':
                with open('time/Blood/client_cnn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pathmnist_2nn':
                with open('time/Path/client_2nn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pathmnist_cnn':
                with open('time/Path/client_cnn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
        elif dp_mechanism == 'both':
            if model_name == 'pneumoniamnist_2nn':
                with open('time/Pneumonia/client_2nn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pneumoniamnist_cnn':
                with open('time/Pneumonia/client_cnn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'bloodmnist_2nn':
                with open('time/Blood/client_2nn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'bloodmnist_cnn':
                with open('time/Blood/client_cnn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pathmnist_2nn':
                with open('time/Path/client_2nn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))
            if model_name == 'pathmnist_cnn':
                with open('time/Path/client_cnn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, times))

        # 测传输数据大小
        Transmission = sys.getsizeof(Net.state_dict())
        if dp_mechanism == 'Gaussian':
            if model_name == 'pneumoniamnist_2nn':
                with open('transmission/Pneumonia/client_2nn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pneumoniamnist_cnn':
                with open('transmission/Pneumonia/client_cnn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'bloodmnist_2nn':
                with open('transmission/Blood/client_2nn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'bloodmnist_cnn':
                with open('transmission/Blood/client_cnn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pathmnist_2nn':
                with open('transmission/Path/client_2nn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pathmnist_cnn':
                with open('transmission/Path/client_cnn_LDP.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
        elif dp_mechanism == 'no_dp':
            if model_name == 'pneumoniamnist_2nn':
                with open('transmission/Pneumonia/client_2nn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pneumoniamnist_cnn':
                with open('transmission/Pneumonia/client_cnn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'bloodmnist_2nn':
                with open('transmission/Blood/client_2nn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'bloodmnist_cnn':
                with open('transmission/Blood/client_cnn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pathmnist_2nn':
                with open('transmission/Path/client_2nn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pathmnist_cnn':
                with open('transmission/Path/client_cnn_FL.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
        elif dp_mechanism == 'both':
            if model_name == 'pneumoniamnist_2nn':
                with open('transmission/Pneumonia/client_2nn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pneumoniamnist_cnn':
                with open('transmission/Pneumonia/client_cnn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'bloodmnist_2nn':
                with open('transmission/Blood/client_2nn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'bloodmnist_cnn':
                with open('transmission/Blood/client_cnn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pathmnist_2nn':
                with open('transmission/Path/client_2nn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
            if model_name == 'pathmnist_cnn':
                with open('transmission/Path/client_cnn_both.txt', 'a') as f:
                    f.write('%d %.3f\n' % (now_num_comm, Transmission))
        '''
        if self.attack == 'MIA':
            print("START STATS ON OVERFITTING WITH MNIST : ", config.statistics.epoch_value)
            res_precision = np.zeros(len(config.statistics.epoch_value))
            res_recall = np.zeros(len(config.statistics.epoch_value))
            res_accuracy = np.zeros(len(config.statistics.epoch_value))
            # res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
            # res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
            if model_name == 'mnist_cnn' or model_name == 'mnist_2nn':
                res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 10))
                res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 10))
                res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 10))
            elif model_name == 'bloodmnist_cnn' or model_name == 'bloodmnist_2nn':
                res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 8))
                res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 8))
                res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 8))
            elif model_name == 'pneumoniamnist_cnn' or model_name == 'pneumoniamnist_2nn':
                res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 2))
                res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 2))
                res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 2))
            elif model_name == 'pathmnist_cnn' or model_name == 'pathmnist_2nn':
                res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 9))
                res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 9))
                res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 9))
            for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
                config.set_subkey("learning", "epochs", value_to_test)
                if model_name == 'mnist_cnn' or model_name == 'mnist_2nn':
                    print("train on mnist with model")
                    precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_mnist(config, path, index_value_to_test, data_train_target, np.array(X), np.array(Y), np.array(C), dp_mechanism, self.dp_epsilon, self.dp_delta, self.rate)
                elif model_name == 'bloodmnist_cnn' or model_name == 'bloodmnist_2nn':
                    print("train on bloodmnist with model")
                    precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_Bloodmnist(config, path, index_value_to_test, data_train_target, np.array(X), np.array(Y), np.array(C), dp_mechanism, self.dp_epsilon, self.dp_delta, self.rate)
                elif model_name == 'pneumoniamnist_cnn' or model_name == 'pneumoniamnist_2nn':
                    print("train on pneumoniamnist with model")
                    precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_Pneumoniamnist(config, path, index_value_to_test, data_train_target, np.array(X), np.array(Y), np.array(C), dp_mechanism, self.dp_epsilon, self.dp_delta, self.rate)
                elif model_name == 'pathmnist_cnn' or model_name == 'pathmnist_2nn':
                    print("train on pathmnist with model")
                    precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = experience_Pathmnist(config, path, index_value_to_test, data_train_target, np.array(X), np.array(Y), np.array(C), dp_mechanism, self.dp_epsilon, self.dp_delta, self.rate)
                res_precision[index_value_to_test] = precision_general
                res_recall[index_value_to_test] = recall_general
                res_accuracy[index_value_to_test] = accuracy_general
                # res_best_acc_target[index_value_to_test] = best_acc_target
                # res_best_acc_shadows[index_value_to_test] = best_acc_shadows
                res_precision_per_class[index_value_to_test] = precision_per_class
                res_recall_per_class[index_value_to_test] = recall_per_class
                res_accuracy_per_class[index_value_to_test] = accuracy_per_class
                np.save(path + "/res_precision.npy", res_precision)
                np.save(path + "/res_recall.npy", res_recall)
                np.save(path + "/res_accuracy.npy", res_accuracy)
                # np.save(path + "/res_best_acc_target.npy", res_best_acc_target)
                np.save(path + "/res_recall_per_class.npy", res_recall_per_class)
                np.save(path + "/res_precision_per_class.npy", res_precision_per_class)
                np.save(path + "/res_accuracy_per_class.npy", res_accuracy_per_class)
                # np.save(path + "/res_best_acc_shadows.npy", res_best_acc_shadows)
        elif self.attack == 'iDLG':
            if model_name == 'mnist_cnn' or model_name == 'mnist_2nn':
                num_classes = 10
            elif model_name == 'bloodmnist_cnn' or model_name == 'bloodmnist_2nn':
                num_classes = 8
            elif model_name == 'pneumoniamnist_cnn' or model_name == 'pneumoniamnist_2nn':
                num_classes = 2
            elif model_name == 'pathmnist_cnn' or model_name == 'pathmnist_2nn':
                num_classes = 9
            model_copy = copy.deepcopy(Net)
            self.iDLG(model_copy, iDLG_data, iDLG_label, model_name, num_classes, now_num_comm, dp_mechanism)
            print("Implement DLG")
        else : print("Without Attack")


        return Net.state_dict()

    def local_val(self):
        pass

    def clip_gradients(self, net, dp_mechanism):
        if dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            # length = 0
            # for k, v in net.named_parameters():
            #     length += 1
            # print("length:", length)
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(1) / self.dp_clip)
                # print(v)
                # break
        elif dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(2) / self.dp_clip)
                # print('k: {}'.format(k))
                # print('###########')
                # print('v: {}'.format(v))
        elif dp_mechanism == 'CDP':
            # Gaussian use 2 norm
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(2) / self.dp_clip)
        elif dp_mechanism == 'both':
            # split parameter
            length = 0
            for k, v in net.named_parameters():
                length += 1
            length_rate = int(length * self.rate)
            length_front = 0
            for k, v in net.named_parameters():
                length_front += 1
                v.grad /= max(1, v.grad.norm(2) / self.dp_clip)
                if length_front >= length_rate:
                    break
            for k, v in net.named_parameters():
                if length_rate != 0:
                   length_rate -= 1
                   continue
                v.grad /= max(1, v.grad.norm(1) / self.dp_clip)

    def add_noise(self, net, dp_mechanism):
        sensitivity = cal_sensitivity(0.01, self.dp_clip, len(self.train_ds))  #写死了要改

        #sensitivity = cal_sensitivity(0.01, self.dp_clip, 10)  # 写死了要改
        if dp_mechanism == 'Laplace':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.dev)
                    v += noise

        elif dp_mechanism == 'Gaussian':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.dev)
                    v += noise

        elif dp_mechanism == 'both':
            with torch.no_grad():
                # add split noise
                length = 0
                for k, v in net.named_parameters():
                    length += 1
                length_rate = int(length * self.rate)
                length_front = 0
                for k, v in net.named_parameters():
                    length_front += 1
                    # noise = Gaussian_Simple(epsilon=self.dp_epsilon * self.rate, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.dev)
                    v += noise
                    if length_front >= length_rate:
                       break
                for k, v in net.named_parameters():
                    if length_rate != 0:
                       length_rate -= 1
                       continue
                    # noise = Laplace(epsilon=self.dp_epsilon * (1 - self.rate), sensitivity=sensitivity, size=v.shape)
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.dev)
                    v += noise

    def weights_init(self, m):
        try:
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
        except Exception:
            print('warning: failed in weights_init for %s.weight' % m._get_name())
        try:
            if hasattr(m, "bias"):
                m.bias.data.uniform_(-0.5, 0.5)
        except Exception:
            print('warning: failed in weights_init for %s.bias' % m._get_name())
    def iDLGclip_gradients(self, original_dy_dx,dp_mechanism):
        if dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            # length = 0
            # for k, v in net.named_parameters():
            #     length += 1
            # print("length:", length)
            for i in range(len(original_dy_dx)):
                original_dy_dx[i] /= max(1, original_dy_dx[i].norm(1) / self.dp_clip)
                # print(v)
                # break
        elif dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            for i in range(len(original_dy_dx)):
                original_dy_dx[i] /= max(1, original_dy_dx[i].norm(2) / self.dp_clip)
                # print('k: {}'.format(k))
                # print('###########')
                # print('v: {}'.format(v))
        elif dp_mechanism == 'CDP':
            # Gaussian use 2 norm, CDP use gaussian
            for i in range(len(original_dy_dx)):
                original_dy_dx[i] /= max(1, original_dy_dx[i].norm(2) / self.dp_clip)
        elif dp_mechanism == 'both':
            # split parameter
            length = 0
            for i in range(len(original_dy_dx)):
                length += 1
            length_rate = int(length * self.rate)
            length_front = 0
            for i in range(len(original_dy_dx)):
                length_front += 1
                original_dy_dx[i] /= max(1, original_dy_dx[i].norm(2) / self.dp_clip)
                if length_front >= length_rate:
                    break
            for i in range(len(original_dy_dx)):
                if length_rate != 0:
                   length_rate -= 1
                   continue
                original_dy_dx[i] /= max(1, original_dy_dx[i].norm(1) / self.dp_clip)

    def iDLGadd_noise(self, original_dy_dx, dp_mechanism):
        sensitivity = cal_sensitivity(0.01, self.dp_clip, len(self.train_ds))  # 写死了要改

        # sensitivity = cal_sensitivity(0.01, self.dp_clip, 10)  # 写死了要改
        if dp_mechanism == 'Laplace':
            for i in range(len(original_dy_dx)):
                noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=original_dy_dx[i].shape)
                noise = torch.from_numpy(noise).to(self.dev)
                original_dy_dx[i] += noise

        elif dp_mechanism == 'Gaussian':
            for i in range(len(original_dy_dx)):
                noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity,
                                            size=original_dy_dx[i].shape)
                noise = torch.from_numpy(noise).to(self.dev)
                original_dy_dx[i] += noise

        elif dp_mechanism == 'both':
            with torch.no_grad():
                # add split noise
                length = 0
                for i in range(len(original_dy_dx)):
                    length += 1
                length_rate = int(length * self.rate)
                length_front = 0
                for i in range(len(original_dy_dx)):
                    length_front += 1
                    # noise = Gaussian_Simple(epsilon=self.dp_epsilon * self.rate, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity,
                                            size=original_dy_dx[i].shape)
                    noise = torch.from_numpy(noise).to(self.dev)
                    original_dy_dx[i] += noise
                    if length_front >= length_rate:
                        break
                for i in range(len(original_dy_dx)):
                    if length_rate != 0:
                        length_rate -= 1
                        continue
                    # noise = Laplace(epsilon=self.dp_epsilon * (1 - self.rate), sensitivity=sensitivity, size=v.shape)
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=original_dy_dx[i].shape)
                    noise = torch.from_numpy(noise).to(self.dev)
                    original_dy_dx[i] += noise

    def iDLG(self, Net, data, label, model_name, num_classes, now_num_comm, dp_mechanism):
        root_path = '.'
        dataset = model_name
        # data_path = os.path.join(root_path, 'data').replace('\\', '/')
        save_path = os.path.join(root_path, 'results/iDLG_%s' % dataset).replace('\\', '/')
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        num_dummy = 1
        # lr = 0.02  2nn
        lr = 0.02
        Iteration = 300
        net = Net
        net.load_state_dict(torch.load('model.pth'))
        # net.apply(self.weights_init)
        for method in ['iDLG']:
        # for method in ['DLG', 'iDLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = now_num_comm               # 取出打乱后的第一个位置index
                imidx_list.append(idx)
                # tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = data    # 取出真实元素的tensor送如cuda
                # tmp_datum = tmp_datum.view(1, *tmp_datum.size())

                # tmp_label = torch.Tensor([dst[idx][1]]).long.to(device)
                tmp_label = label   # 把该元素标签送入cuda

                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)   # 张量连接
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)


            # compute original gradient
            out = Net(gt_data)             # 输出置信度
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, Net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            if dp_mechanism != 'no_dp':
                self.iDLGclip_gradients(original_dy_dx, dp_mechanism)
                if dp_mechanism != 'CDP':
                    self.iDLGadd_noise(original_dy_dx, dp_mechanism)

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)    # 输入假数据假标签
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            print('lr =', lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = Net(dummy_data)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                        # dummy_loss = criterion(pred, gt_label)
                    elif method == 'iDLG':
                        # dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                        dummy_loss = criterion(pred, label_pred)

                    dummy_dy_dx = torch.autograd.grad(dummy_loss, Net.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data-gt_data)**2).item())


                if iters % int(Iteration / 30) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))
                    # history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    shape3 = 1
                    if model_name == 'bloodmnist_cnn' or model_name == 'bloodmnist_2nn' or model_name == 'pathmnist_cnn' or model_name == 'pathmnist_2nn':
                        shape3 = 3
                    history.append([(dummy_data[imidx].reshape(28, 28, shape3).cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        # plt.imshow(tp(gt_data[imidx].cpu()))
                        plt.imshow((gt_data[imidx].reshape(28, 28, shape3).cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            # plt.imshow(history[i][imidx])
                            plt.imshow(((history[i][imidx].detach().numpy()) * 255).astype(np.uint8))
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, now_num_comm))
                            plt.close()

                    # if current_loss < 0.000001: # converge
                    #     break

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                # label_DLG = torch.tensor([item.cpu().detach().numpy() for item in torch.argmax(dummy_label, dim=-1)]).cuda()

                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses



        print('imidx_list:', imidx_list)
        # print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
        # print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
        # print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)

        print('loss_iDLG:', loss_iDLG[-1])
        print('mse_iDLG:', mse_iDLG[-1])

        print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_iDLG:', label_iDLG)


        if model_name == 'pathmnist_2nn':
            np.savetxt("iDLGpathmnist_2nn.txt", mse_iDLG)  # 保存文件
            with open('results/iDLG_pathmnist_2nn/mse.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if model_name == 'bloodmnist_2nn':
            with open('results/iDLG_bloodmnist_2nn/mse.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if model_name == 'pneumoniamnist_2nn':
            with open('results/iDLG_pneumoniamnist_2nn/mse.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if model_name == 'pathmnist_cnn':
            with open('results/iDLG_pathmnist_cnn/mse.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if model_name == 'bloodmnist_cnn':
            with open('results/iDLG_bloodmnist_cnn/mse.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if model_name == 'pneumoniamnist_cnn':
            with open('results/iDLG_pneumoniamnist_cnn/mse.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if dp_mechanism == 'Laplace' and model_name == 'bloodmnist_2nn':
            with open('results/iDLG_bloodmnist_2nn/mseLaplace.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if dp_mechanism == 'Gaussian' and model_name == 'bloodmnist_2nn':
            with open('results/iDLG_bloodmnist_2nn/mseGaussian.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if dp_mechanism == 'both' and model_name == 'bloodmnist_2nn':
            with open('results/iDLG_bloodmnist_2nn/mseboth.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        ######################################################################
        if dp_mechanism == 'Laplace' and model_name == 'pneumoniamnist_2nn':
            with open('results/iDLG_pneumoniamnist_2nn/mseLaplace.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if dp_mechanism == 'Gaussian' and model_name == 'pneumoniamnist_2nn':
            with open('results/iDLG_pneumoniamnist_2nn/mseGaussian.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if dp_mechanism == 'both' and model_name == 'pneumoniamnist_2nn':
            with open('results/iDLG_pneumoniamnist_2nn/mseboth.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if dp_mechanism == 'Laplace' and model_name == 'pathmnist_2nn':
            with open('results/iDLG_pathmnist_2nn/mseLaplace.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if dp_mechanism == 'Gaussian' and model_name == 'pathmnist_2nn':
            with open('results/iDLG_pathmnist_2nn/mseGaussian.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        if dp_mechanism == 'both' and model_name == 'pathmnist_2nn':
            with open('results/iDLG_pathmnist_2nn/mseboth.txt', 'a') as f:
                f.write('%.3f\n' % (min(mse_iDLG)))
        print('----------------------\n\n')


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.TestTensorDataset = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        if self.data_set_name == 'mnist':
            mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

            test_data = torch.tensor(mnistDataSet.test_data)
            test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
            # self.TestTensorDataset = TensorDataset(test_data, test_label)
            self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=10, shuffle=False, drop_last=True)

            train_data = mnistDataSet.train_data
            train_label = mnistDataSet.train_label

            shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
            shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
            for i in range(self.num_of_clients):
                shards_id1 = shards_id[i * 2]
                shards_id2 = shards_id[i * 2 + 1]
                data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_label = np.argmax(local_label, axis=1)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
        elif self.data_set_name == 'pneumoniamnist':
            mnistDataSet = GetDataSetpneumoniamnist(self.data_set_name, self.is_iid)

            test_data = torch.tensor(mnistDataSet.test_data)
            # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)

            test_label = []
            for i in range(mnistDataSet.test_data_size):
                test_label.append(mnistDataSet.test_label[i][0])
            test_label = torch.tensor(test_label, dtype=torch.int64)

            self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=10, shuffle=False,
                                               drop_last=True)
            self.test_data_loaders = DataLoader(TensorDataset(test_data, test_label), batch_size=600, shuffle=False,
                                                drop_last=True)
            train_data = mnistDataSet.train_data
            train_label = mnistDataSet.train_label

            shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
            shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
            for i in range(self.num_of_clients):
                shards_id1 = shards_id[i * 2]
                shards_id2 = shards_id[i * 2 + 1]
                data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                local_data = np.vstack((data_shards1, data_shards2))
                local_label = []
                for j in range(shard_size):
                    local_label.append(label_shards1[j][0])
                for j in range(shard_size):
                    local_label.append(label_shards2[j][0])
                # local_label = np.argmax(local_label, axis=1)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label, dtype=torch.int64)),
                                 self.dev)
                self.clients_set['client{}'.format(i)] = someone
        elif self.data_set_name == 'bloodmnist':
            mnistDataSet = GetDataSetbloodmnist(self.data_set_name, self.is_iid)

            test_data = torch.tensor(mnistDataSet.test_data)
            # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)

            test_label = []
            for i in range(mnistDataSet.test_data_size):
                test_label.append(mnistDataSet.test_label[i][0])
            test_label = torch.tensor(test_label, dtype=torch.int64)

            self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=10, shuffle=False,
                                               drop_last=True)
            self.test_data_loaders = DataLoader(TensorDataset(test_data, test_label), batch_size=600, shuffle=False,
                                                drop_last=True)
            train_data = mnistDataSet.train_data
            train_label = mnistDataSet.train_label

            shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
            shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
            for i in range(self.num_of_clients):
                shards_id1 = shards_id[i * 2]
                shards_id2 = shards_id[i * 2 + 1]
                data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                local_data = np.vstack((data_shards1, data_shards2))
                local_label = []
                for j in range(shard_size):
                    local_label.append(label_shards1[j][0])
                for j in range(shard_size):
                    local_label.append(label_shards2[j][0])
                # local_label = np.argmax(local_label, axis=1)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label, dtype=torch.int64)),
                                 self.dev)
                self.clients_set['client{}'.format(i)] = someone
        elif self.data_set_name == 'pathmnist':
            mnistDataSet = GetDataSetpathmnist(self.data_set_name, self.is_iid)

            test_data = torch.tensor(mnistDataSet.test_data)
            # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)

            test_label = []
            for i in range(mnistDataSet.test_data_size):
                test_label.append(mnistDataSet.test_label[i][0])
            test_label = torch.tensor(test_label, dtype=torch.int64)

            self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=10, shuffle=False,
                                               drop_last=True)
            self.test_data_loaders = DataLoader(TensorDataset(test_data, test_label), batch_size=600, shuffle=False,
                                                drop_last=True)
            train_data = mnistDataSet.train_data
            train_label = mnistDataSet.train_label

            shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
            shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
            for i in range(self.num_of_clients):
                shards_id1 = shards_id[i * 2]
                shards_id2 = shards_id[i * 2 + 1]
                data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                local_data = np.vstack((data_shards1, data_shards2))
                local_label = []
                for j in range(shard_size):
                    local_label.append(label_shards1[j][0])
                for j in range(shard_size):
                    local_label.append(label_shards2[j][0])
                # local_label = np.argmax(local_label, axis=1)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label, dtype=torch.int64)),
                                 self.dev)
                self.clients_set['client{}'.format(i)] = someone
if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN, PneumoniaMnist_2NN, PneumoniaMnist_CNN, BloodMnist_2NN, BloodMnist_CNN, PathMnist_2NN, PathMnist_CNN, ResNet18, ViT
from clients import ClientsGroup, client
from dp_mechanism import *

import matplotlib.pyplot as plt
import time


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=1, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='bloodmnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')
parser.add_argument('-dpm', '--dp_mechanism', type=str, default='no_dp', help='run what kind of differential privarcy')
parser.add_argument('-ak', '--attack_mechanism', type=str, default='no_attack', help='run what kind of attack')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__
    # list = ['no_dp', 'Gaussian', 'both', 'Laplace', 'CDP']
    # list = ['Gaussian']

    list = ['no_dp', 'Gaussian', 'both']
    for x in range(len(list)):
        args['dp_mechanism'] = list[x]
        print(args['dp_mechanism'])
        start = time.time()
        Correct_list = []  # 建立数组保存准确率
        Loss_list = []  # 建立数组保存loss
        test_mkdir(args['save_path'])

        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        net = None
        if args['model_name'] == 'mnist_2nn':
            net = Mnist_2NN()
            myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'mnist_cnn':
            # net = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            #             dim=64, depth=6, heads=8, mlp_dim=128)
            net = Mnist_CNN()
            myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'pneumoniamnist_2nn':
            net = PneumoniaMnist_2NN()
            myClients = ClientsGroup('pneumoniamnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'pneumoniamnist_cnn':
            net = PneumoniaMnist_CNN()
            myClients = ClientsGroup('pneumoniamnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'bloodmnist_2nn':
            net = BloodMnist_2NN()
            myClients = ClientsGroup('bloodmnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'bloodmnist_cnn':
            net = BloodMnist_CNN()
            myClients = ClientsGroup('bloodmnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'pathmnist_2nn':
            net = PathMnist_2NN()
            myClients = ClientsGroup('pathmnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'pathmnist_cnn':
            net = PathMnist_CNN()
            myClients = ClientsGroup('pathmnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'mnist_resnet18':
            net = ResNet18()
            myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
        elif args['model_name'] == 'pneumoniamnist_resnet18':
            net = ResNet18()
            myClients = ClientsGroup('pneumoniamnist', args['IID'], args['num_of_clients'], dev)

        if torch.cuda.device_count() >= 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = torch.nn.DataParallel(net)
        net = net.to(dev)

        loss_func = F.cross_entropy
        opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

        testDataLoader = myClients.test_data_loader
        # testTensorDataset = myClients.TestTensorDataset

        num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

        global_parameters = {}
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
        The_first_client = 0
        for i in range(args['num_comm']):
            print("communicate round {}".format(i+1))

            order = np.random.permutation(args['num_of_clients'])
            clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

            attack = 'no_attack'
            sum_parameters = None
            The_first_client = 0
            time3 = time.time()
            for client in tqdm(clients_in_comm):
                if The_first_client == 0:
                    if args['attack_mechanism'] == 'MIA':
                        if i+1 == 2 or i+1 == 4 or i+1 == 8 or i+1 == 16 or i+1 == 32:
                            attack = 'MIA'
                        The_first_client = 1
                    if args['attack_mechanism'] == 'iDLG':
                        attack = 'iDLG'
                        The_first_client = 1
                else:
                    attack = 'no_attack'
                local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                             loss_func, opti, global_parameters, args['dp_mechanism'], args['num_comm'], i + 1, client, attack, args['model_name'])
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] + local_parameters[var]
            for var in global_parameters:
                global_parameters[var] = (sum_parameters[var] / num_in_comm)
            time4 = time.time()
            times = time4 - time3
            if args['dp_mechanism'] == 'Gaussian':
                if args['model_name'] == 'pneumoniamnist_2nn':
                    with open('time/server/Pneumonia/server_2nn_LDP20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pneumoniamnist_cnn':
                    with open('time/server/Pneumonia/server_cnn_LDP20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'bloodmnist_2nn':
                    with open('time/server/Blood/server_2nn_LDP20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'bloodmnist_cnn':
                    with open('time/server/Blood/server_cnn_LDP20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pathmnist_2nn':
                    with open('time/server/Path/server_2nn_LDP20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pathmnist_cnn':
                    with open('time/server/Path/server_cnn_LDP20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
            if args['dp_mechanism'] == 'no_dp':
                if args['model_name'] == 'pneumoniamnist_2nn':
                    with open('time/server/Pneumonia/server_2nn_FL20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pneumoniamnist_cnn':
                    with open('time/server/Pneumonia/server_cnn_FL20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'bloodmnist_2nn':
                    with open('time/server/Blood/server_2nn_FL20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'bloodmnist_cnn':
                    with open('time/server/Blood/server_cnn_FL20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pathmnist_2nn':
                    with open('time/server/Path/server_2nn_FL20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pathmnist_cnn':
                    with open('time/server/Path/server_cnn_FL20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
            if args['dp_mechanism'] == 'both':
                if args['model_name'] == 'pneumoniamnist_2nn':
                    with open('time/server/Pneumonia/server_2nn_both20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pneumoniamnist_cnn':
                    with open('time/server/Pneumonia/server_cnn_both20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'bloodmnist_2nn':
                    with open('time/server/Blood/server_2nn_both20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'bloodmnist_cnn':
                    with open('time/server/Blood/server_cnn_both20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pathmnist_2nn':
                    with open('time/server/Path/server_2nn_both20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))
                if args['model_name'] == 'pathmnist_cnn':
                    with open('time/server/Path/server_cnn_both20.txt', 'a') as f:
                        f.write('%d %.3f\n' % (i + 1, times))

            if args['dp_mechanism'] == 'CDP':
                net.load_state_dict(global_parameters, strict=True)  # 加载全局参数
                sensitivity = cal_sensitivity(0.01, 0.01, 0.01)  # 写死了要改
                with torch.no_grad():
                    for k, v in net.named_parameters():
                        noise = Gaussian_Simple(epsilon=0, delta=1e-5, sensitivity=sensitivity,
                                                size=v.shape)
                        noise = torch.from_numpy(noise).to(dev)
                        v += noise
                print("CDP")
            with torch.no_grad():
                if (i + 1) % args['val_freq'] == 0:
                    net.load_state_dict(global_parameters, strict=True)
                    sum_accu = 0
                    running_loss = 0.0
                    loss_list = []
                    num = 0
                    for data, label in testDataLoader:
                        # data = data.reshape(10, 1, 28, 28)
                        # print(data.size())
                        data, label = data.to(dev), label.to(dev)
                        preds = net(data)
                        loss = loss_func(preds, label)
                        preds = torch.argmax(preds, dim=1)
                        sum_accu += (preds == label).float().mean()
                        num += 1
                        running_loss += loss.item()
                        loss_list.append(loss.item())
                    print('accuracy: {}'.format(sum_accu / num))
                    Correct_list.append(100. * ((sum_accu.cpu().numpy()) / num))  # 保存准确率
                    Loss_list.append(running_loss / num)
                    if args['IID'] == 0:
                        if args['dp_mechanism'] == 'no_dp':
                            with open('result/nonidd/Path/0.51/accurate_records_FL.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, (sum_accu.cpu().numpy()) / num))
                                print('*************************************')
                            with open('result/nonidd/Path/0.51/loss_records_FL.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, running_loss / num))
                        elif args['dp_mechanism'] == 'CDP':
                            with open('result/nonidd/Path/0.51/accurate_records_CDP.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, (sum_accu.cpu().numpy()) / num))
                            with open('result/nonidd/Path/0.51/loss_records_CDP.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, running_loss / num))
                        elif args['dp_mechanism'] == 'Gaussian':
                            with open('result/nonidd/Path/0.51/accurate_records_Gaussian.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, (sum_accu.cpu().numpy()) / num))
                            with open('result/nonidd/Path/0.51/loss_records_Gaussian.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, running_loss / num))
                        elif args['dp_mechanism'] == 'Laplace':
                            with open('result/nonidd/Path/0.51/accurate_records_Laplace.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, (sum_accu.cpu().numpy()) / num))
                            with open('result/nonidd/Path/0.51/loss_records_Laplace.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, running_loss / num))
                        # elif args['dp_mechanism'] == 'both':
                        #     with open('result/nonidd/Path/1/accurate_records_RandomHybrid.txt', 'a') as f:
                        #         f.write('%d %.3f\n' % (i + 1, (sum_accu.cpu().numpy()) / num))
                        #     with open('result/nonidd/Path/1/loss_records_RandomHybrid.txt', 'a') as f:
                        #         f.write('%d %.3f\n' % (i + 1, running_loss / num))
                        elif args['dp_mechanism'] == 'both':
                            with open('result/nonidd/Path/0.51/accurate_records_OptimizedHybrid.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, (sum_accu.cpu().numpy()) / num))
                            with open('result/nonidd/Path/0.51/loss_records_OptimizedHybrid.txt', 'a') as f:
                                f.write('%d %.3f\n' % (i + 1, running_loss / num))

            if (i + 1) % args['save_freq'] == 0:
                torch.save(net, os.path.join(args['save_path'],
                                             '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                    i, args['epoch'],
                                                                                                    args['batchsize'],
                                                                                                    args['learning_rate'],
                                                                                                    args['num_of_clients'],
                                                                                                    args['cfraction'])))

        print("tag3")
        end = time.time()
        print('Time:', end - start)
        # x2 = range(1, int(args['num_comm'] / args['val_freq']) + 1)
        #
        # y2 = Correct_list
        #
        # plt.subplot(2, 1, 1)
        # plt.plot(x2, y2, '.-')
        # plt.xlabel('Accuracy vs. epoches')
        # plt.ylabel('Test Accuracy')
        # plt.show()
        # #
        # x1 = range(0, int(args['num_comm'] / args['val_freq']))
        #
        # y1 = Loss_list
        #
        # plt.subplot(2, 1, 1)
        # plt.plot(x1, y1, '.-')
        # plt.xlabel('Loss vs. epoches')
        # plt.ylabel('Test Loss')
        # plt.show()

    # 保存
    # if args['dpm'] == 'Laplace':
    #     np.save('accuracy_list_laplace_0.5.txt', y2)
    #     np.save('loss_list_laplace_0.5.txt', y1)
    # else:
    #     pass
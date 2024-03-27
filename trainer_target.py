import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from dp_mechanism import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model_target(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, dp_algorithm = '', dp_epsilon =0, dp_delta=0, rate = 0.5):
    print("DATASET SIZE", dataset_sizes)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    retunr_value_train = np.zeros((4, num_epochs))

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        X = []
        Y = []
        C = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()  将这一句调到for循环结束前
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                inputs, labels = data.to(device), target.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if epoch == num_epochs-1:
                        for out in outputs.cpu().detach().numpy():
                            X.append(out)
                            if phase == "train":
                                Y.append(1)
                            else:
                                Y.append(0)
                        for cla in labels.cpu().detach().numpy():
                            C.append(cla)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                retunr_value_train[0][epoch] = epoch_loss
                retunr_value_train[1][epoch] = epoch_acc
            else:
                retunr_value_train[2][epoch] = epoch_loss
                retunr_value_train[3][epoch] = epoch_acc



            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
        if dp_algorithm == 'Laplace':
            model = add_noise(model, 'Laplace', dp_epsilon, dp_delta, rate)
        elif dp_algorithm == 'Gaussian':
            model = add_noise(model, 'Gaussian', dp_epsilon, dp_delta, rate)
        elif dp_algorithm == 'both':
            model = add_noise(model, 'Gaussian', dp_epsilon, dp_delta, rate)
        #print()
    print('Train accuracy')
    print(retunr_value_train[1])
    print('Val accuracy')
    print(retunr_value_train[3])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    # load best model weights
    # model.load_state_dict(best_model_wts)

    # for k, v in model.named_parameters():
    #     print("without noise v")
    #     print(v)
    #     break


    # for k, v in model.named_parameters():
    #     print("within noise v")
    #     print(v)
    #     break
    return model, retunr_value_train, np.array(X), np.array(Y), np.array(C)

def add_noise(net, dp_mechanism, dp_epsilon, dp_delta, rate):
    sensitivity = cal_sensitivity(0.01, 1e-2, 0)  #写死了要改

    #sensitivity = cal_sensitivity(0.01, self.dp_clip, 10)  # 写死了要改
    if dp_mechanism == 'Laplace':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Laplace(epsilon=dp_epsilon, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise

    elif dp_mechanism == 'Gaussian':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Gaussian_Simple(epsilon=dp_epsilon, delta=dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise
    #
    elif dp_mechanism == 'both':
        with torch.no_grad():
            # add split noise
            length = 0
            for k, v in net.named_parameters():
                length += 1
            length_rate = int(length * rate)
            length_front = 0
            for k, v in net.named_parameters():
                length_front += 1
                # noise = Gaussian_Simple(epsilon=self.dp_epsilon * self.rate, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = Gaussian_Simple(epsilon=dp_epsilon, delta=dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise
                if length_front >= length_rate:
                    break
            for k, v in net.named_parameters():
                if length_rate != 0:
                    length_rate -= 1
                    continue
                # noise = Laplace(epsilon=self.dp_epsilon * (1 - self.rate), sensitivity=sensitivity, size=v.shape)
                noise = Laplace(epsilon=dp_epsilon, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise

    return net

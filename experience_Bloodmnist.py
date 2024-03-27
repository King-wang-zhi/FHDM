import torch
import torch.nn as nn
import torch.optim as optim
from dataloaders import *
from utils import config
import numpy as np
from model import *
from Models import *
from torch.optim import lr_scheduler
from trainer import *
from trainer_target import *
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import lightgbm as lgb

import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



def experience_Bloodmnist(config, path, param, data_train_attack, X, Y, C, dp_mechanism, dp_epsilon, dp_delta, rate):
    print("START MNIST")

    use_cuda = config.general.use_cuda and torch.cuda.is_available()
    torch.manual_seed(config.general.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    bloodmnist_data = np.load('./data/bloodmnist.npz')

    train_images = bloodmnist_data['train_images']
    test_images = bloodmnist_data['test_images']
    train_labels = bloodmnist_data['train_labels']
    test_labels = bloodmnist_data['test_labels']

    train_images = train_images.reshape(11959, 28, 28, 3)
    test_images = test_images.reshape(3421, 28, 28, 3)

    train_images = train_images.reshape(train_images.shape[0],
                                        train_images.shape[1] * train_images.shape[2] * train_images.shape[3])
    test_images = test_images.reshape(test_images.shape[0],
                                      test_images.shape[1] * test_images.shape[2] * test_images.shape[3])

    train_images = train_images.astype(np.float32)
    train_images = np.multiply(train_images, 1.0 / 255.0)
    test_images = test_images.astype(np.float32)
    test_images = np.multiply(test_images, 1.0 / 255.0)

    train_labels = train_labels.squeeze(axis=1)
    test_labels = test_labels.squeeze(axis=1)

    train_images = torch.from_numpy(train_images)
    test_images = torch.from_numpy(test_images)
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)

    data_train_target = train_images, train_labels
    data_test_target = test_images, test_labels


    train_loader_target = DataLoader(TensorDataset(train_images, train_labels), batch_size=config.learning.batch_size, shuffle=True, drop_last=True)
    test_loader_target = DataLoader(TensorDataset(test_images, test_labels), batch_size=config.learning.batch_size, shuffle=True, drop_last=True)
    dataloaders_target = {"train": train_loader_target, "val": test_loader_target}
    dataset_sizes_target = {"train": train_images.size(0), "val": test_images.size(0)}
    # To is for shadow model

    print("START TRAINING SHADOW MODEL")
    all_shadow_models = []
    all_dataloaders_shadow = []
    data_train_set = []
    label_train_set = []
    class_train_set = []
    for num_model_sahdow in range(config.general.number_shadow_model):
        criterion = nn.CrossEntropyLoss()

        model_shadow = BloodMnist_CNN().to(device)
        optimizer = optim.SGD(model_shadow.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor, gamma=config.learning.decrease_lr_every)
        model_shadow, best_acc_sh, data_train_set_unit, label_train_set_unit, class_train_set_unit = train_model_target(model_shadow, criterion, optimizer, exp_lr_scheduler, dataloaders_target, dataset_sizes_target, num_epochs=config.learning.epochs, dp_algorithm=dp_mechanism, dp_epsilon=dp_epsilon, dp_delta=dp_delta ,rate = rate)
        data_train_set.append(data_train_set_unit)
        label_train_set.append(label_train_set_unit)
        class_train_set.append(class_train_set_unit)
        np.save(path + "/res_train_shadow_"+str(num_model_sahdow)+"_"+str(param)+".npy", best_acc_sh)
        all_shadow_models.append(model_shadow)

    print("START GETTING DATASET ATTACK MODEL")
    data_train_set = np.concatenate(data_train_set)
    label_train_set = np.concatenate(label_train_set)
    class_train_set = np.concatenate(class_train_set)
    #data_test_set, label_test_set, class_test_set = get_data_for_final_eval([model_target], [dataloaders_target], device)
    #data_train_set, label_train_set, class_train_set = get_data_for_final_eval(all_shadow_models, all_dataloaders_shadow, device)
    data_train_set, label_train_set, class_train_set = shuffle(data_train_set, label_train_set, class_train_set, random_state=config.general.seed)
    print(X.shape)
    print(Y.shape)
    print(C.shape)
    test_size_ = len(Y)
    X_data = np.random.randn(test_size_, 8)
    Y_data = np.random.randint(0, 1, size = [1, test_size_])   # 范围[)
    C_data = np.random.randint(0, 8, size = [1, test_size_])
    X = np.append(X, X_data, axis=0)
    Y = np.append(Y, Y_data)
    C = np.append(C, C_data)
    data_test_set, label_test_set, class_test_set = shuffle(X, Y, C, random_state=config.general.seed)
    print("Taille dataset train", len(label_train_set))
    print("Taille dataset test", len(label_test_set))
    print("START FITTING ATTACK MODEL")
    # params = {'device': 'gpu',
    # 'gpu_platform_id': 0,
    # 'gpu_device_id': 0}
    model = lgb.LGBMClassifier(objective='binary', reg_lambda=config.learning.ml.reg_lambd, n_estimators=config.learning.ml.n_estimators)

    # data_train_set, label_train_set = (torch.from_numpy(data_train_set)).to(device), (torch.from_numpy(label_train_set)).to(device)
    model.fit(data_train_set, label_train_set)

    y_pred_lgbm = model.predict(data_test_set)
    precision_general, recall_general, _, _ = precision_recall_fscore_support(y_pred=y_pred_lgbm, y_true=label_test_set, average = "macro")
    accuracy_general = accuracy_score(y_true=label_test_set, y_pred=y_pred_lgbm)
    precision_per_class, recall_per_class, accuracy_per_class = [], [], []
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
    ]
    for idx_class, classe in enumerate(classes):
        all_index_class = np.where(class_test_set == idx_class)
        precision, recall, _, _ = precision_recall_fscore_support(y_pred=y_pred_lgbm[all_index_class], y_true=label_test_set[all_index_class], average = "macro")
        accuracy = accuracy_score(y_true=label_test_set[all_index_class], y_pred=y_pred_lgbm[all_index_class])
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        accuracy_per_class.append(accuracy)
    print("END MNIST")
    return (precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class)

import numpy as np
import gzip
import os
import platform
import pickle


class GetDataSetpathmnist(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'pathmnist':
            self.mnistDataSetConstruct(isIID)
        else:
            pass


    def mnistDataSetConstruct(self, isIID):
        # train pathmnist
        pathmnist_data = np.load('./data/pathmnist.npz')

        train_images = pathmnist_data['train_images']
        test_images = pathmnist_data['test_images']
        train_labels = pathmnist_data['train_labels']
        test_labels = pathmnist_data['test_labels']

        #train pathmnist
        train_images = train_images.reshape(89996, 28, 28, 3)
        test_images = test_images.reshape(7180, 28, 28, 3)


        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 3
        assert test_images.shape[3] == 3
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2] * train_images.shape[3])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2] * test_images.shape[3])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)   # 产生数组0到n-1
            np.random.shuffle(order)                  # 打乱数组
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # labels = np.argmax(train_labels, axis=1)  # 按行方向返回最大值索引
            labels = train_labels.reshape(-1)
            order = np.argsort(labels)                # 返回从小到大排序元素对应的索引
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        # 数据分配方式是否是独立同分布的



        self.test_data = test_images
        self.test_label = test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSetpathmnist('pathmnist', True) # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])


import numpy as np
import functools
import chainer.links as L
import chainer.functions as F
from collections import defaultdict
import nutszebra_chainer


class VGG_A(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(VGG_A, self).__init__()
        modules = []
        modules += [('conv1', L.Convolution2D(3, 64, 3, 1, 1))]
        modules += [('conv2', L.Convolution2D(64, 128, 3, 1, 1))]
        modules += [('conv3_1', L.Convolution2D(128, 256, 3, 1, 1))]
        modules += [('conv3_2', L.Convolution2D(256, 256, 3, 1, 1))]
        modules += [('conv4_1', L.Convolution2D(256, 512, 3, 1, 1))]
        modules += [('conv4_2', L.Convolution2D(512, 512, 3, 1, 1))]
        modules += [('conv5_1', L.Convolution2D(512, 512, 3, 1, 1))]
        modules += [('conv5_2', L.Convolution2D(512, 512, 3, 1, 1))]
        modules += [('fc1', L.Convolution2D(512, 4096, 7, 1, 0))]
        modules += [('fc2', L.Convolution2D(4096, 4096, 1, 1, 0))]
        modules += [('fc3', L.Convolution2D(4096, category_num, 1, 1, 0))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'VGG_A_{}'.format(category_num)

    @staticmethod
    def _count_conv_parameters(conv):
        return functools.reduce(lambda a, b: a * b, conv.W.data.shape)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += VGG_A._count_conv_parameters(link)
        return count

    def weight_initialization(self):
        for name, link in self.modules:
            self[name].W.data = self.weight_relu_initialization(link)
            self[name].b.data = self.bias_initialization(link, constant=0)

    def __call__(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=(2, 2), stride=(2, 2), pad=(0, 0))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=(2, 2), stride=(2, 2), pad=(0, 0))
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.max_pooling_2d(h, ksize=(2, 2), stride=(2, 2), pad=(0, 0))
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.max_pooling_2d(h, ksize=(2, 2), stride=(2, 2), pad=(0, 0))
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.max_pooling_2d(h, ksize=(2, 2), stride=(2, 2), pad=(0, 0))
        h = F.dropout(h, ratio=0.5, train=train)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, ratio=0.5, train=train)
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        num, categories, y, x = h.data.shape
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy

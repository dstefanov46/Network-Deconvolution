'''
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, AveragePooling2D, Reshape

from deconv import *

class Block(tf.keras.Model):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, deconv=None):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        if deconv:
            self.deconv=True
            self.conv1 = deconv(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=True,n_iter=1)
            self.conv3 = deconv(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True,block=in_planes,n_iter=1)

            self.shortcut = tf.keras.Sequential()
            if stride == 1 and in_planes != out_planes:
                self.shortcut = tf.keras.Sequential()
                self.shortcut.add(deconv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True))
        else:
            self.deconv = False
            self.conv1 = Conv2D(planes, kernel_size=1, use_bias=False)
            self.bn1 = BatchNormalization(axis=3)
            self.conv2 = Conv2D(planes, kernel_size=3, strides=stride, padding='same', groups=planes,
                                use_bias=False)
            self.bn2 = BatchNormalization(axis=3)
            self.conv3 = Conv2D(out_planes, kernel_size=1, use_bias=False)
            self.bn3 = BatchNormalization(axis=3)

            self.shortcut = tf.keras.Sequential()
            if stride == 1 and in_planes != out_planes:
                self.shortcut = tf.keras.Sequential()
                self.shortcut.add(Conv2D(out_planes, kernel_size=1, use_bias=False))
                self.shortcut.add(BatchNormalization(axis=3))

    def call(self, x):
        if self.deconv:
            out = tf.keras.activations.relu(self.conv1(x))
            out = tf.keras.activations.relu(self.conv2(out))
            out = self.conv3(out)
        else:
            out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
            out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(tf.keras.Model):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, deconv=None,delinear=None):
        super(MobileNetV2, self).__init__()
        if deconv:
            self.deconv=True
            self.conv1 = deconv(3, 32, kernel_size=3, stride=1, padding=1, bias=True,freeze=True,n_iter=10)
            self.conv2 = deconv(320, 1280, kernel_size=1, stride=1, padding=0, bias=True)
            
        else:
            self.conv1 = Conv2D(32, kernel_size=3, padding='same', use_bias=False)
            self.bn1 = BatchNormalization(axis=3)
            self.conv2 = Conv2D(1280, kernel_size=1, use_bias=False)
            self.bn2 = BatchNormalization(axis=3)

        self.model_layers = self._make_layers(in_planes=32, deconv=deconv)
        
        if delinear:
            self.linear = delinear(1280, num_classes)
        else:
            self.linear = Dense(num_classes, activation=tf.nn.softmax)


    def _make_layers(self, in_planes, deconv=None):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, deconv=deconv))
                in_planes = out_planes
        blocks = tf.keras.Sequential()
        for layer in layers:
            blocks.add(layer)
        return blocks

    def call(self, x):
        if hasattr(self,'bn1'):
            out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        else:
            out = tf.keras.activations.relu(self.conv1(x))

        out = self.model_layers(out)
        if hasattr(self,'bn2'):
            out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        else:
            out = tf.keras.activations.relu(self.conv2(out))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        if hasattr(self, 'deconv1'):
            out = self.deconv1(out)
        out = AveragePooling2D((4, 4))(out)
        out = tf.expand_dims(tf.reshape(out,
                                        (tf.shape(out)[0], tf.shape(out)[1] * tf.shape(out)[2] * tf.shape(out)[3])),
                             axis=1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = tf.random.uniform((2,32,32,3))
    y = net(x)
    print(y.shape)

# test()

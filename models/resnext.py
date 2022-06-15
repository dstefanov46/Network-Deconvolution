
'''
See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, AveragePooling2D, Reshape

# from deconv import *

class Block(tf.keras.Model):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1, deconv=None):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        if not deconv:
            self.deconv=False
            self.conv1 = Conv2D(group_width, kernel_size=1, use_bias=False)
            self.bn1 = BatchNormalization(axis=1)
            self.conv2 = Conv2D(group_width, kernel_size=3, strides=stride, padding='same', groups=cardinality, use_bias=False)
            self.bn2 = BatchNormalization(axis=1)
            self.conv3 = Conv2D(self.expansion*group_width, kernel_size=1, use_bias=False)
            self.bn3 = BatchNormalization(axis=1)

            self.shortcut = tf.keras.Sequential()
            if stride != 1 or in_planes != self.expansion*group_width:
                self.shortcut = tf.keras.Sequential()
                self.shortcut.add(Conv2D(self.expansion*group_width, kernel_size=1, strides=stride, use_bias=False))
                self.shortcut.add(BatchNormalization(axis=1))
        else:
            self.deconv=True
            
            self.conv1 = deconv(in_planes, group_width, kernel_size=1, bias=True)
            self.conv2 = deconv(group_width, group_width, kernel_size=3, stride=stride, padding=1, bias=True,groups=cardinality,n_iter=3)
            self.conv3 = deconv(group_width, self.expansion*group_width, kernel_size=1, bias=True)
            
            self.shortcut = tf.keras.Sequential()
            if stride != 1 or in_planes != self.expansion*group_width:
                self.shortcut = tf.keras.Sequential()
                self.shortcut.add(deconv(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=True))

    def call(self, x):
        if self.deconv:
            out = tf.keras.activations.relu(self.conv1(x))
            out = tf.keras.activations.relu(self.conv2(out))
            out = self.conv3(out)
        else:
            out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
            out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = tf.keras.activations.relu(out)
        return out


class ResNeXt(tf.keras.Model):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10, deconv=None, delinear=None):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        if deconv is None:
            self.conv1 = Conv2D(64, kernel_size=1, use_bias=False)
            self.bn1 = BatchNormalization(axis=1)
        else:
            self.conv1=deconv(3, 64, kernel_size=1, bias=True,freeze=True,n_iter=10)
        self.layer1 = self._make_layer(num_blocks[0], 1, deconv=deconv)
        self.layer2 = self._make_layer(num_blocks[1], 2, deconv=deconv)
        self.layer3 = self._make_layer(num_blocks[2], 2, deconv=deconv)
        if delinear:
            self.linear = delinear(cardinality*bottleneck_width*8, num_classes)
        else:
            self.linear = Dense(num_classes, activation=tf.nn.softmax)   

    def _make_layer(self, num_blocks, stride, deconv):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride, deconv=deconv))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        blocks = tf.keras.Sequential()
        for layer in layers:
            blocks.add(layer)
        return blocks

    def call(self, x):
        if hasattr(self,'bn1'):
            out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        else:
            out = tf.keras.activations.relu(self.conv1(x))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if hasattr(self, 'deconv1'):
            out = self.deconv1(out)
        out = AveragePooling2D((8, 8))(out)
        out = tf.expand_dims(tf.reshape(out, (tf.shape(out)[0], tf.shape(out)[1] * tf.shape(out)[2] * tf.shape(out)[3])),
                             axis=1)
        out = self.linear(out)
        return out


def ResNeXt29_32x4d(num_classes, deconv, delinear):
    return ResNeXt(num_blocks=[3,3,3],
                   cardinality=32,
                   bottleneck_width=4,
                   num_classes=num_classes,
                   deconv=deconv,
                   delinear=delinear)


def test_resnext():
    net = ResNeXt29_32x4d(5, None, True)
    x = tf.random.uniform((1,32,32,3))
    y = net.call(x)
    print(y.shape)
    print(y)

# test_resnext()

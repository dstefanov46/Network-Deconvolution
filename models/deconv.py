import time
import math

import tensorflow as tf
from tensorflow.keras.layers import Dot, Layer, Dense

# Iteratively solve for the inverse sqrt of a matrix
def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA = tf.norm(A, ord=2, axis=[-2,-1], keepdims=False)
    Y = A / normA
    I = tf.eye(dim, dtype=A.dtype)
    Z = tf.eye(dim, dtype=A.dtype)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z@Y)
        Y = Y@T
        Z = T@Z
    A_isqrt = Z / tf.math.sqrt(normA)
    return A_isqrt

def isqrt_newton_schulz_autograd_batch(A, num_iters):
    batchSize,dim,_ = A.shape
    norm_A = tf.expand_dims(tf.norm(tf.reshape(A, (batchSize, -1)), ord=2, axis=1, keepdims=True), axis=0)
    Y = tf.math.divide(A, norm_A)
    I = tf.broadcast_to(tf.eye(dim, dtype=A.dtype), A.shape)
    Z = tf.broadcast_to(tf.eye(dim, dtype=A.dtype), A.shape)

    for i in range(num_iters):
        T = 0.5*(3.0*I - Dot(axes=[2, 1])([Z, Y]))
        Y = Dot(axes=[2, 1])([Y, T])
        Z = Dot(axes=[2, 1])([T, Z])

    A_isqrt = Z / tf.math.sqrt(norm_A)

    return A_isqrt


class Delinear(tf.keras.Model):

    def __init__(self, in_features, out_features, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=512):
        super(Delinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.add_weight(shape=(out_features, in_features), initializer=tf.keras.initializers.HeUniform())
        if bias:
            self.bias = self.add_weight(shape=(out_features,), initializer=tf.keras.initializers.HeUniform())
        else:
            self.bias = None

        if block > in_features:
            block = in_features
        else:
            if in_features % block != 0:
                block = math.gcd(block, in_features)
                print('block size set to:', block)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.running_mean = tf.Variable(tf.zeros((self.block,)), trainable=False)
        self.running_deconv = tf.Variable(tf.eye(self.block), trainable=False)

    def call(self, input):

        if self.trainable:

            # 1. reshape
            X = tf.reshape(input, (-1, self.block))

            # 2. subtract mean
            X_mean = tf.math.reduce_mean(X, 0)
            X = X - tf.expand_dims(X_mean, 0)
            self.running_mean.assign(self.running_mean * (1 - self.momentum))
            self.running_mean.assign(self.running_mean + X_mean * self.momentum)

            # 3. calculate COV, COV^(-0.5), then deconv
            # Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Id = tf.eye(X.shape[1])
            Cov = tf.transpose(X) @ X / tf.cast(tf.shape(X)[0], tf.float32)
            Cov += self.eps * Id
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            # track stats for evaluation
            self.running_deconv.assign(self.running_deconv * (1 - self.momentum))
            self.running_deconv.assign(self.running_deconv + deconv * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        w = tf.reshape(self.weight, (-1, self.block)) @ deconv
        if self.bias is None:
            b = - tf.math.reduce_sum(tf.reshape(w @ (tf.expand_dims(X_mean, 1)), (tf.shape(self.weight)[0], -1)), 1)
        else:
            b = self.bias - tf.math.reduce_sum(tf.reshape(w @ (tf.expand_dims(X_mean, 1)),
                                                          (tf.shape(self.weight)[0], -1)), 1)

        w = tf.reshape(w, self.weight.shape)
        return tf.linalg.matmul(input, tf.linalg.matrix_transpose(w)) + self.bias
        

class FastDeconv(tf.keras.layers.Conv2D):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,freeze=False,freeze_iter=100):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps

        super(FastDeconv, self).__init__(out_channels, kernel_size)

        if block > in_channels:
            block = in_channels
        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            # grouped conv
            block=in_channels//groups

        self.block=block

        self.num_features = kernel_size**2 *block
        if groups==1:
            self.running_mean = tf.Variable(tf.zeros((self.num_features,)), trainable=False)
            self.running_deconv = tf.Variable(tf.eye(self.num_features), trainable=False)
        else:
            self.running_mean = tf.Variable(tf.zeros((kernel_size ** 2 * in_channels,)), trainable=False)
            self.running_deconv = tf.Variable(tf.repeat(tf.eye(self.num_features), in_channels // block, axis=0), trainable=False)

        self.sampling_stride=sampling_stride*stride
        self.counter=0
        self.freeze_iter=freeze_iter
        self.freeze=freeze

    def call(self, x):
        N, C, H, W = x.shape
        B = self.block
        frozen=self.freeze and (self.counter>self.freeze_iter)
        if self.trainable:
            self.counter+=1
            self.counter %= (self.freeze_iter * 10)

        if self.trainable and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:
                X = tf.image.extract_patches(x, 
                                             sizes=[1, self.kernel_size[0], self.kernel_size[0], 1],
                                             strides=[1,1,1,1],
                                             rates=[1,1,1,1],
                                             padding='same')
                X = tf.transpose(X, perm=[0, 2, 1, 3])
            else:
                #channel wise
                X = tf.reshape(tf.transpose(x, perm=[0, 2, 3, 1]), (-1, C))[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                X = tf.reshape(tf.transpose(tf.reshape(X, (-1, self.num_features, C // B)), perm=[0, 2, 1]),
                               (-1, self.num_features))
            else:
                X = tf.reshape(X, (-1, X.shape[-1]))

            # 2. subtract mean
            X_mean = tf.math.reduce_mean(X, 0)
            X = X - tf.expand_dims(X_mean, 0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups==1:
                #Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id = tf.eye(X.shape[1])
                Cov = tf.transpose(X) @ X / tf.cast(tf.shape(X)[0], tf.float32)
                Cov += self.eps * Id
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            else:
                X = tf.transpose(tf.reshape(X, (-1, self.groups, self.num_features)), perm=[1, 0, 2])
                Id = tf.stack([tf.eye(self.num_features)] * self.groups)
                Cov = tf.transpose(X, perm=[0, 2, 1]) @ X * 1. / X.shape[1] + self.eps * Id
                deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)

            if self.trainable:
                # track stats for evaluation
                self.running_mean.assign(self.running_mean * (1 - self.momentum))
                self.running_mean.assign(self.running_mean + X_mean * self.momentum)
                self.running_deconv.assign(self.running_deconv * (1 - self.momentum))
                self.running_deconv.assign(self.running_deconv + deconv * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        #4. X * deconv * conv = X * (deconv * conv)
        if self.groups==1:
            w = tf.reshape(tf.transpose(tf.reshape(self.weights[0], (-1, self.num_features, C // B)),
                                        perm=[0, 2, 1]), (-1,self.num_features)) @ deconv
            b = self.bias - tf.math.reduce_sum(tf.reshape(w @ tf.expand_dims(X_mean, 1),
                                                          (tf.shape(self.weights[0])[0], -1)), 1)
            w = tf.transpose(tf.reshape(w, (-1, C // B, self.num_features)), perm=[0, 2, 1])
        else:
            w = tf.reshape(self.weights[0], (C//B, -1, self.num_features)) @ deconv
            b = self.bias - tf.reshape(w @ tf.reshape(X_mean, (-1, self.num_features, 1)), self.bias.shape)

        w = tf.reshape(w, tf.shape(self.weights[0]))
        x = tf.nn.conv2d(x, w, self.stride, 'SAME', data_format='NCHW', dilations=self.dilation)

        return x



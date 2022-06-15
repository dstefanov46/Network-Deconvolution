from __future__ import print_function
from functools import partial
import csv

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow import keras
from tensorflow.keras.layers import Normalization

from models.deconv import *
from arg_parser import *


if __name__ == '__main__':

    args = parse_args()

    if args.deconv:
        args.deconv = partial(FastDeconv,
                              bias=args.bias,
                              eps=args.eps,
                              n_iter=args.deconv_iter,
                              block=args.block,
                              sampling_stride=args.stride)
    else:
        args.deconv = None

    if args.delinear:
        args.channel_deconv = None
        if args.block_fc > 0:
            args.delinear = partial(Delinear, block=args.block_fc, eps=args.eps, n_iter=args.deconv_iter)
        else:
            args.delinear = None

    tf.random.set_seed(args.seed)

    # Data
    print('==> Preparing data..')

    if args.dataset == 'cifar10':
        args.in_planes = 3
        args.input_size = 32

        print("| Preparing CIFAR-10 dataset...")
        args.num_outputs = 10
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_train = tf.transpose(tf.image.random_crop(X_train, size=X_train.shape, seed=args.seed), perm=[0,3,1,2]).numpy()
        X_train = tf.image.random_flip_left_right(X_train, seed=args.seed).numpy() / 255
        X_train = Normalization(mean=(0.4914, 0.4822, 0.4465), variance=np.square((0.2023, 0.1994, 0.2010)), axis=1)(X_train).numpy()
        y_train = tf.one_hot(y_train, args.num_outputs).numpy()
        X_test = tf.transpose(X_test, perm=[0,3,1,2]) / 255
        X_test = Normalization(mean=(0.4914, 0.4822, 0.4465), variance=np.square((0.2023, 0.1994, 0.2010)), axis=1)(X_test).numpy()
        y_test = tf.one_hot(y_test, args.num_outputs).numpy()
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print(np.max(X_train), np.min(X_train), np.unique(y_train), np.max(X_test), np.min(X_test), np.unique(y_test))

    elif (args.dataset == 'cifar100'):
        args.in_planes = 3
        args.input_size = 32

        print("| Preparing CIFAR-100 dataset...")
        args.num_outputs = 100
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
        X_train = tf.transpose(tf.image.random_crop(X_train, size=X_train.shape, seed=args.seed), perm=[0,3,1,2]).numpy()
        X_train = tf.image.random_flip_left_right(X_train, seed=args.seed).numpy() / 255
        X_train = Normalization(mean=(0.5071, 0.4865, 0.4409), variance=np.square((0.2009, 0.1984, 0.2023)), axis=1)(X_train).numpy()
        y_train = tf.one_hot(y_train, args.num_outputs).numpy()
        X_test = tf.transpose(X_test, perm=[0,3,1,2]) / 255
        X_test = Normalization(mean=(0.5071, 0.4865, 0.4409), variance=np.square((0.2009, 0.1984, 0.2023)), axis=1)(X_test).numpy()
        y_test = tf.one_hot(y_test, args.num_outputs).numpy()

    print('==> Building model..')

    if args.deconv:
        args.batchnorm=False
        print('************ Batch norm disabled when deconv is used. ************')

    if args.arch == 'resnext':
        from models.resnext import *
        net = ResNeXt29_32x4d(num_classes=args.num_outputs,
                                deconv=args.deconv,
                                delinear=args.delinear)
    if args.arch == 'mobilev2':
        from models.mobilenetv2 import *
        net = MobileNetV2(num_classes=args.num_outputs,
                            deconv=args.deconv,
                            delinear=args.delinear)

    inputs = tf.keras.Input(shape=(32, 32, 3))
    outputs = net(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    lr = args.lr
    total_steps = math.ceil(len(X_train) / args.batch_size) * args.epochs
    args.current_scheduler = tf.keras.optimizers.schedules.CosineDecay(args.lr,
                                                                       total_steps,
                                                                       alpha=0)

    if args.optimizer == 'SGD':
        args.current_optimizer = tf.keras.optimizers.SGD(learning_rate=args.current_scheduler,
                                                         momentum=args.momentum)
    elif args.optimizer == 'Adam':
        args.current_optimizer = tf.keras.optimizers.Adam(learning_rate=args.current_scheduler)

    model.compile(optimizer=args.current_optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    if args.deconv == False:
        log_name = 'BN'  
    else:
        log_name = 'ND'
    csv_logger = keras.callbacks.CSVLogger(f'{args.arch}_{log_name}_{args.optimizer}_training.log', append=True)
    model.fit(x=X_train,
              y=y_train,
              workers=args.num_workers,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(X_test, y_test),
              callbacks=[csv_logger])

    # Training
    print(args)

    params = np.sum([np.prod(weight.get_shape()) for weight in net.trainable_weights])
    print(params, 'trainable parameters in the network.\n')
    print('Training finished successfully.')


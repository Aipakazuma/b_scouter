from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import argparse


WIDTH = 320
HEIGHT = 200


def define_model():
    # モデル構築
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='SAME', input_shape=(WIDTH, HEIGHT, 3)))
    model.add(MaxPooling2D((2, 2), strides=(1, 1), padding='VALID'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    return model


def train_op(model):
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def train(args):
    num_classes = 2
    data_0_obj = Data(data_dir_path=os.path.join(args.data_dir, 'train', 'favarit'))
    data_0 = data_0_obj.data_sets
    label_0 = np.zeros(len(data_0))

    data_1_obj = Data(data_dir_path=os.path.join(args.data_dir, 'train', 'not_favarit'))
    data_1 = data_1_obj.data_sets[:len(data_0) + 1]
    label_1 = np.ones(len(data_1))

    data, labels = data_0 + data_1, label_0 + label_1
    X_train = np.asarray(data)
    Y_train = labels

    data_0_obj = Data(data_dir_path=os.path.join(args.data_dir, 'test', 'favarit'))
    data_0 = data_0_obj.data_sets
    label_0 = np.zeros(len(data_0))

    data_1_obj = Data(data_dir_path=os.path.join(args.data_dir, 'test', 'not_favarit'))
    data_1 = data_1_obj.data_sets[:len(data_0) + 1]
    label_1 = np.ones(len(data_1))

    data, labels = data_0 + data_1, label_0 + label_1
    X_test = np.asarray(data)
    Y_test = labels

    # Convert class vectors to binary class matrices.
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    tensorboard = TensorBoard()
    callbacks = [tensorboard]
    model = train_op(define_model())
    print(model.summary())

    model.fit(X_train, Y_train, epochs=args.n_epoch,
              verbose=2, batch_size=args.batch_size,
              validation_data=(X_test, Y_test), callbacks=callbacks)

    # モデルを保存
    model.save_weights('%s.hdf5' % ('test'))


def test(args):
    num_classes = 2
    data_0_obj = Data(data_dir_path=os.path.join(args.data_dir, 'test', 'favarit'))
    data_0 = data_0_obj.data_sets
    label_0 = np.zeros(len(data_0))

    data_1_obj = Data(data_dir_path=os.path.join(args.data_dir, 'test', 'not_favarit'))
    data_1 = data_1_obj.data_sets[:len(data_0) + 1]
    label_1 = np.ones(len(data_1))

    data, labels = data_0 + data_1, label_0 + label_1
    X_test = np.array(data)
    Y_test = labels
    Y_test = to_categorical(Y_test, num_classes)

    model = train_op(define_model())
    model.load_weights(args.weights_file)
    predict = model.predict(x_test, batch_size=args.batch_size)
    print(predict)


def argument():
    parser = argparse.ArgumentParser(description='Training and test command.')
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')
    # training argument
    train = subparsers.add_parser('train')
    # 必要だったら追加
    # train.add_argument('--weight-file', type=int, help='weight file.')
    train.add_argument('--batch_size', type=int, help='batch number.')
    train.add_argument('--n_epoch', type=int, help='epoch number.')
    train.add_argument('--data_dir', type=int, help='data directory')
    # test argument
    test = subparsers.add_parser('test')
    test.add_argument('--batch_size', type=int, help='batch number.')
    test.add_argument('--weights_file', type=str, help='weight fiel.')
    test.add_argument('--data_dir', type=int, help='data directory')
    return parser


if __name__ == '__main__':
    parser = argument()
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
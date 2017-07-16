import tensorflow as tf
from model import inference, loss, train
from data import Data
import argparse
import numpy as np
import os
import sys
from time import time
from datetime import datetime
import traceback


def training(args):
    width = 120
    height = 169
    channel = 3
    data_obj = Data(data_dir_path=args.data_dir)
    data = data_obj.data_sets
    labels = np.zeros(len(data), np.int64)
    labels[:int(len(data)/2)] = 1

    with tf.Graph().as_default():
        input_images = tf.constant(data)
        input_labels = tf.constant(labels)

        images = tf.decode_raw(input_images, tf.uint8)
        images = tf.reshape(images, (len(data), height, width, channel))
        images = tf.cast(images, tf.float32)
        images.set_shape([len(data), height, width, channel])

        slice_images, slice_labels = tf.train.slice_input_producer(
            [images, input_labels])
        X_batch, Y_batch = tf.train.batch(
            [slice_images, slice_labels], batch_size=args.batch_size)

        global_step = tf.Variable(0, trainable=False)
        use_fp16 = False
        checkpoint_path = os.path.join(args.ckpt_dir, 'model.ckpt')

        logits = inference(X_batch, args.batch_size, use_fp16)
        total_loss = loss(logits, Y_batch)
        train_op = train(total_loss, global_step, args.batch_size)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True))

        with tf.Session(config=config) as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter(checkpoint_path, sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(
                        sess=sess, coord=coord)

            try:
                # training
                for step in range(args.max_step):
                    start_time = time()
                    _, loss_value = sess.run([train_op, total_loss])
                    duration = time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 10 == 0:
                        num_examples_per_step = args.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value,
                                            examples_per_sec, sec_per_batch))

                    if step % 100 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == args.max_step:
                        saver.save(sess, checkpoint_path, global_step=step)
            except:
                print(traceback.format_exc())

            finally:
                coord.request_stop()
                coord.join(threads)


def test(args):
    pass


def argument():
    parser = argparse.ArgumentParser(description='Training and test command.')
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # training argument
    train = subparsers.add_parser('train')
    train.add_argument('--ckpt_dir', type=str, help='check point directry.')
    train.add_argument('--data_dir', type=str, help='data images directry.')
    train.add_argument('--max_step', type=int, help='step number.')
    train.add_argument('--batch_size', type=int, help='batch number.')

    # test argument
    test = subparsers.add_parser('test')
    test.add_argument('--predict_image_path', type=str, help='predict image path.')
    test.add_argument('--max_step', type=int, help='step number.')

    return parser


if __name__ == '__main__':
    parser = argument()
    args = parser.parse_args()

    if args.command:
        training(args)
    elif args.command:
        test(args)

import tensorflow as tf
import re


def _variable_on_cpu(name, shape, initializer, use_fp16):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_fp16):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
        use_fp16)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    TOWER_NAME='tower'
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


# TODO: この関数でエラーが発生するのでどうにかする
def visualize_hidden_layer_output(summary_name, layer_output,
                                  image_size, n_feature):
    """学習中のconvolutionのoutput画像をtensorboardへ表示."""
    with tf.variable_scope('output'):
        v = tf.slice(layer_output, [0, 0, 0, 0], [1, -1, -1, -1])
        v = tf.reshape(v, (image_size[0], image_size[1], n_feature))
        v = tf.transpose(v, (2, 0, 1))
        v = tf.reshape(v, (-1, image_size[0], image_size[1], 1))
        tf.summary.image('visualize', v, n_feature)


def _add_loss_summaries(total_loss, logger=True):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    if logger:
        for l in losses + [total_loss]:
          # Name each loss as '(raw)' and name the moving average version of the loss
          # as the original loss name.
          tf.summary.scalar(l.op.name +' (raw)', l)
          tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op


def inference(x, batch_size, use_fp16, logger=True):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0,
                                             use_fp16=use_fp16)
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), use_fp16)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        if logger:
            _activation_summary(conv1)
            # visualize_hidden_layer_output('conv1/output', conv1, tf.shape(conv1), 64)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0,
                                             use_fp16=use_fp16)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), use_fp16)
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        if logger:
            _activation_summary(conv2)
            # visualize_hidden_layer_output('conv1/output', conv2, tf.shape(conv2), 64)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    with tf.variable_scope('fc3') as scope:
        pool2_shape = pool2.get_shape()
        reshape = tf.reshape(pool2, [batch_size, (pool2_shape[1] * pool2_shape[2] * pool2_shape[3]).value])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 512],
                                              stddev=0.04, wd=0.004, use_fp16=use_fp16)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1), use_fp16)
        fc3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        if logger:
            _activation_summary(fc3)

    with tf.variable_scope('fc4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[512, 128],
                                              stddev=0.04, wd=0.004, use_fp16=use_fp16)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
        fc4 = tf.nn.relu(tf.matmul(fc3, weights) + biases, name=scope.name)
        if logger:
            _activation_summary(fc4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', shape=[128, 2],
                                              stddev=1/128.0, wd=0.0, use_fp16=use_fp16)
        biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0), use_fp16)
        softmax_linear = tf.add(tf.matmul(fc4, weights), biases, name=scope.name)
        if logger:
            _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step, batch_size, logger=True):
    # Variables that affect learning rate.
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1280
    NUM_EPOCHS_PER_DECAY = 350.0
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  
    # Decay the learning rate exponentially based on the number of steps.
    INITIAL_LEARNING_RATE = 0.1
    LEARNING_RATE_DECAY_FACTOR = 0.1
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    if logger:
        tf.summary.scalar('learning_rate', lr)
  
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss, logger)
  
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
  
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if logger:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
  
    # Add histograms for gradients.
    if logger:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op

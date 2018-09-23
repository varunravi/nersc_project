import tensorflow as tf
import numpy as np
import os
import horovod.tensorflow as hvd


print(tf.__version__)

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
# y_train=y_train.reshape(y_train.shape[0])

# x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_train.shape[3])
# y_test=y_test.reshape(y_test.shape[0])

x_train = np.load('/global/homes/v/vrav2/LBNL_key_items/notebooks/deeplens/test/x_train.npy')
y_train = np.load('/global/homes/v/vrav2/LBNL_key_items/notebooks/deeplens/test/y_train.npy')
x_test = np.load('/global/homes/v/vrav2/LBNL_key_items/notebooks/deeplens/test/x_test.npy')
y_test = np.load('/global/homes/v/vrav2/LBNL_key_items/notebooks/deeplens/test/y_test.npy')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[3], x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[3], x_test.shape[1])

num_class = 10
num_steps_total=1000000

def conv2D(layer, 
           ft_size, 
           name, 
           ksize=1, 
           strides=[1, 1, 1, 1], 
           padding="SAME", 
           initializer=tf.contrib.layers.xavier_initializer(),
           dtype=tf.float32
          ):
  
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  
        w = tf.get_variable(name='w', shape=[ksize, ksize, layer.shape[3], ft_size], dtype=dtype, initializer=initializer)
        b = tf.get_variable(name='b', shape=[ft_size], dtype=dtype, initializer=tf.zeros_initializer())

        layer = tf.nn.conv2d(input=layer, filter=w, strides=strides, padding=padding)
        layer = tf.add(layer, b)

    return layer

def respath_fn(respath, in_layer, name):

    with tf.variable_scope("residual_path"+name, reuse=tf.AUTO_REUSE):      

        new_shape = [respath.get_shape().as_list()[0], np.amax([in_layer.get_shape().as_list()[1], respath.get_shape().as_list()[1]]), np.amax([in_layer.get_shape().as_list()[2], respath.get_shape().as_list()[2]]), np.amax([in_layer.get_shape().as_list()[3], respath.get_shape().as_list()[3]])]

        res_padding = [[0, 0], [0, new_shape[1]-respath.get_shape().as_list()[1]], [0, new_shape[2]-respath.get_shape().as_list()[2]], [0, new_shape[3]-respath.get_shape().as_list()[3]]]
        in_padding = [[0, 0], [0, new_shape[1]-in_layer.get_shape().as_list()[1]], [0, new_shape[2]-in_layer.get_shape().as_list()[2]], [0, new_shape[3]-in_layer.get_shape().as_list()[3]]]

        layer = tf.add(tf.pad(in_layer, in_padding, 'CONSTANT'), tf.pad(respath, res_padding, 'CONSTANT'))

        return layer
    
tf.reset_default_graph()

# graph
HEIGHT = x_train.shape[1]
WIDTH = x_train.shape[2]
CHANNELS = x_train.shape[3]
NUM_CLASS = num_class

x = tf.placeholder(dtype=tf.float32, shape=[None, WIDTH, HEIGHT, CHANNELS], name='x')
y = tf.placeholder(dtype=tf.int64, shape=None, name='y')
batch = tf.placeholder(dtype=tf.int64, shape=None, name='batch')

with tf.variable_scope("hvd", reuse=tf.AUTO_REUSE):
    hvd.init()

with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
    layer = conv2D(layer=x, ft_size=64, name='_1', ksize=7, strides=[1, 2, 2, 1])

resnet_layer = layer

with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):
    layer = tf.nn.max_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool')
    for i in range(3):
        with tf.variable_scope("block_"+str(i), reuse=tf.AUTO_REUSE):
            layer = conv2D(layer, 64, '_0')
            layer = conv2D(layer, 64, '_1', ksize=3)
            layer = conv2D(layer, 256, '_2')

resnet_layer = respath_fn(resnet_layer, layer, '_0')
layer = resnet_layer

with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE):      
    for i in range(4):
        with tf.variable_scope("block_"+str(i), reuse=tf.AUTO_REUSE):
            layer = conv2D(layer, 128, '_0')
            layer = conv2D(layer, 128, '_1', ksize=3)
            layer = conv2D(layer, 512, '_2')

resnet_layer = respath_fn(resnet_layer, layer, '_1')
layer = resnet_layer
    
with tf.variable_scope("conv4", reuse=tf.AUTO_REUSE): 
    for i in range(6):
        with tf.variable_scope("block_"+str(i), reuse=tf.AUTO_REUSE):
            layer = conv2D(layer, 256, '_0')
            layer = conv2D(layer, 256, '_1', ksize=3)
            layer = conv2D(layer, 1024, '_2')

resnet_layer = respath_fn(resnet_layer, layer, '_2')
layer = resnet_layer

with tf.variable_scope("conv5", reuse=tf.AUTO_REUSE): 
    for i in range(3):
        with tf.variable_scope("block_"+str(i), reuse=tf.AUTO_REUSE):
            layer = conv2D(layer, 512, '_0')
            layer = conv2D(layer, 512, '_1', ksize=3)
            layer = conv2D(layer, 2048, '_2')

with tf.variable_scope("output", reuse=tf.AUTO_REUSE):      
    layer = tf.nn.avg_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='avg_pool')
    #   layer = tf.contrib.layers.fully_connected(layer, 1000)
    layer = tf.contrib.layers.flatten(layer)
    w_loss = tf.get_variable(name='w', shape=[layer.shape[1], NUM_CLASS], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b_loss = tf.get_variable(name='b', shape=[NUM_CLASS], dtype=tf.float32, initializer=tf.zeros_initializer())
    layer = tf.matmul(layer, w_loss) + b_loss

with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, 10), logits=layer))
    optimize = tf.train.AdagradOptimizer(1e-4)
    
    optimize = hvd.DistributedOptimizer(optimize)

    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    hooks.append(tf.train.StopAtStepHook(last_step=num_steps_total))
    
    global_step = tf.train.get_or_create_global_step()
    train_op = optimize.minimize(loss, global_step=global_step)

with tf.variable_scope("accuracy", reuse=tf.AUTO_REUSE):     
    softmax = tf.argmax(tf.nn.softmax(logits=layer, axis=1), axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(softmax, y), tf.float32))*100

if hvd.rank() == 0:
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

config = tf.ConfigProto()
config.intra_op_parallelism_threads =68
config.inter_op_parallelism_threads =4

checkpoint_dir = './tmp/train_logs' if hvd.rank() == 0 else None

with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, hooks=hooks, config=config) as mon_sess:
    train_writer = tf.summary.FileWriter(checkpoint_dir, mon_sess.graph)
    while not mon_sess.should_stop():
        mon_sess.run(train_op, feed_dict={x: x_train[0:5], y: y_train[0:5]})
        if hvd.rank() == 0:
            summary, current_loss, current_global_step = mon_sess.run([merged, loss, global_step], feed_dict={x: x_train[0:5], y: y_train[0:5]})
            print("loss: %f global_step: %d" % (current_loss, current_global_step))

            current_accuracy = mon_sess.run([accuracy], feed_dict={x: x_test[0:5], y: y_test[0:5]})
            print('accuracy: %.2f%%' % current_accuracy[0])
            
            train_writer.add_summary(summary, current_global_step)
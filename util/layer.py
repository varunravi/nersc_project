import tensorflow as tf
import numpy as np

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
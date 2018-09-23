import tensorflow as tf
from util.layer import conv2D, respath_fn
    

def resnet50(
    width, 
    height, 
    channels, 
    batch_size,
    epochs,
    distributed_mode=False):


    # DATA PROCESSING
    x = tf.placeholder(dtype=tf.float32, shape=[None, width, height, channels], name='x')
    y = tf.placeholder(dtype=tf.int64, shape=None, name='y')

    handle = tf.placeholder(dtype=tf.string, shape=[])
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=10000).batch(batch_size).repeat()

    iterator = tf.data.Iterator.from_string_handle(string_handle=handle, output_types=(x.dtype, y.dtype), output_shapes=(x.shape, y.shape))
    
    training_iterator = dataset.make_initializable_iterator()
    validation_iterator = dataset.make_initializable_iterator()
    
    next_element = iterator.get_next()

    # RESNET
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
        
        layer = tf.contrib.layers.flatten(layer)  

        w_fully = tf.get_variable(name='w_fully', shape=[layer.shape[1], 1000], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b_fully = tf.get_variable(name='b_fully', shape=[1000], dtype=tf.float32, initializer=tf.zeros_initializer())
        layer = tf.matmul(layer, w_fully) + b_fully

        w_pred = tf.get_variable(name='w_pred', shape=[layer.shape[1], 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b_pred = tf.get_variable(name='b_pred', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
        layer = tf.matmul(layer, w_pred) + b_pred

    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=layer))
        
        optimize = tf.train.AdagradOptimizer(1e-4)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimize.minimize(loss, global_step=global_step)


    # PREDICTION/STATISTICS
    with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):     
        pred = tf.sigmoid(x=layer)

        precision = tf.metrics.precision(y, pred)
        recall = tf.metrics.recall(y, pred)
        auc = tf.metrics.auc(y, pred)

        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('AUC', auc)

        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()

    return merged, loss, optimize, layer




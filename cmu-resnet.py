import tensorflow as tf
import numpy as np
from util.graph import resnet50

print(tf.__version__)

## LOAD DATA

# x_train = np.load('./deeplens/test/x_train.npy')
# y_train = np.load('./deeplens/test/y_train.npy')
# x_test = np.load('./deeplens/test/x_test.npy')
# y_test = np.load('./deeplens/test/y_test.npy')

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[3], x_train.shape[1])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[3], x_test.shape[1])

x_train = np.random.rand(100, 101, 101, 3)
y_train = np.random.randint(0,2,[100,1])

x_val = np.random.rand(100, 101, 101, 3)
y_val = np.random.randint(0,2,[100,1])

## HYPERPARAMETERS

JOBID = 'cmu_resnet-v1'
CHECKPOINT_DIR = '~/chkpt'
EPOCHS = 20
BATCH_SIZE = 32
TOTAL_STEPS = x_train.shape[0]/BATCH_SIZE*EPOCHS
HEIGHT = x_train.shape[1]
WIDTH = x_train.shape[2]
CHANNELS = x_train.shape[3]

 
# def run_dist():
#     config = tf.ConfigProto()
#     config.intra_op_parallelism_threads = 68
#     config.inter_op_parallelism_threads = 4

                
if __name__ == '__main__':
    
    [merged, loss, optimize, layer] = resnet50(width=WIDTH, height=HEIGHT, channels=CHANNELS, batch_size=BATCH_SIZE, epochs=EPOCHS)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        sess.run(training_iterator.initializer, feed_dict={x:x_train, y:x_val})
        sess.run(validation_iterator.initializer, feed_dict={x:x_test, y:y_val})

        if (tf.train.latest_checkpoint(CHECKPOINT_DIR) != None):
            if(tf.train.checkpoint_exists(tf.train.latest_checkpoint(CHECKPOINT_DIR))):
                saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))        



    
#     if distributed:
#         run_dist()
        













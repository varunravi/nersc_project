import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
from util.graph import cmu_resnet

print(tf.__version__)

x_train = np.load('./deeplens/test/x_train.npy')
y_train = np.load('./deeplens/test/y_train.npy')
x_test = np.load('./deeplens/test/x_test.npy')
y_test = np.load('./deeplens/test/y_test.npy')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[3], x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[3], x_test.shape[1])

JOBID = 'cmu_resnet-v1'
CHPKT = '~/chkpt'
EPOCH = 10
NUM_CLASS = 10
BATCH_SIZE = 32
TOTAL_STEPS = x_train.shape[0]/BATCH_SIZE*EPOCH
HEIGHT = x_train.shape[1]
WIDTH = x_train.shape[2]
CHANNELS = x_train.shape[3]

 
def run_dist():
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 68
    config.inter_op_parallelism_threads = 4

    checkpoint_dir = './train_logs' if hvd.rank() == 0 else None

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

                
if __name__ == '__main__':
    
    g = cmu_resnet(width=101, height=101, channels=3, num_class=10)
    
#     if distributed:
#         run_dist()
        
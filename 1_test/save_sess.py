import tensorflow as tf
import numpy as np
weights = tf.Variable(tf.random_normal([784,20],stddev=0.35),name="weights")

heights = tf.Variable(tf.random_normal([20,20]),name="heights")

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

np.set_printoptions(threshold='nan')
with tf.Session() as sess:
    sess.run(init_op)
    # saver.restore(sess,"/tmp/tensorflow_data/model.ckpt")
    print(weights.eval())

    save_path = saver.save(sess,"/tmp/tensorflow_data/model.ckpt")
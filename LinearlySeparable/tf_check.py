# Verify that tensorflow is working
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# print version
print('TensorFlow version is: ' + str(tf.__version__))

# verify that session works
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
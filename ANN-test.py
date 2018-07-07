import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import sin,sign,tan
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.INFO)

def sigmoidr(x):
    return tf.nn.sigmoid(-x)
def cnn_model_fn(features,labels,mode):
    '''model function for CNN'''
    #input layer
    #input_layer = tf.reshape(features["x"], [-1,28*28*4])
    input_layer = tf.reshape(features["x"], [-1,28*28])
    #dense layers
    dense_layer_1 = tf.layers.dense(input_layer, 28*28, activation=tf.nn.softmax)
    dense_layer_2 = tf.layers.dense(dense_layer_1,16,activation=tf.nn.softmax)
    dense_layer_3 = tf.layers.dense(dense_layer_2,16,activation=tf.nn.softmax)
    #logic layer
    logits = tf.layers.dense(dense_layer_2, 10)


    #predictions
    predictions = {
        "classes":tf.argmax(input=logits,axis=1),
        "probabilities":tf.nn.softmax(logits,name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    #calculate loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)
#    loss = tf.losses.mean_squared_error(tf.reshape(labels,[-1]),tf.reshape(logits,[-1]))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(labels, predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def main(unusedargv):
    #inport data.
    mnist=tf.contrib.learn.datasets.load_dataset("mnist")
    #load data respectively.
    train_data=mnist.train.images
    train_labels=np.asarray(mnist.train.labels,dtype=np.int32)
    eval_data=mnist.test.images
    eval_labels=np.asarray(mnist.test.labels,dtype=np.int32)

    #create an estimator.
    mnist_classifier=tf.estimator.Estimator(cnn_model_fn,model_dir="/tmp/mnist_convnet_model")

    #loggings
    tensors_to_log={"probabilities":"softmax_tensor"}
    logging_hook=tf.train.LoggingTensorHook(tensors_to_log,every_n_iter=50)

    #training
    train_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn,steps=20000,hooks=[logging_hook])

    #evaling
    eval_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results=mnist_classifier.evaluate(eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
    tf.app.run()

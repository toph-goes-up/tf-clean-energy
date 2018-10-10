##########################
## Chris Kinsey         ##
## CS 5890              ##
## IS / Clean Energy    ##
## TF output estimation ##
##########################

import numpy as np
import datetime
import tensorflow as tf
from sklearn.preprocessing import scale
from plot_conf import plot_confusion_matrix


FILE_PATH = "hourly-weather-output.csv"
LEARNING_RATE = 0.1 # Not used when the Adam optimizer is used.
EPOCHS = 500
BATCH_SIZE = 10
N_BINS = "doane"
TOP_K_LOGITS = 3

# Parse data from the CSV
### TODO: Add cyclic encoding for day and hour ###
text = np.loadtxt(open(FILE_PATH, "rb"), delimiter=',', skiprows=1, usecols=range(1, 12),
                  converters={ # Convert the binary date to a string, then to a datetime, then to an integer day of the year
                      1: lambda s: datetime.datetime.strptime(s.decode("ascii"), "%m/%d/%Y").timetuple().tm_yday
                  })

m, n = text.shape

# Set up input and output vectors
x_full = scale(text[:, :-1], axis=0)
y_full = text[:, -1]
hist, bins = np.histogram(y_full, bins = N_BINS)
y_full = np.digitize(y_full, bins, right=True)
y_full = y_full.astype("int32")

# Split into training and test data
x_train = x_full[0:5000]
x_test = x_full[5000:]
y_train = y_full[0:5000]
y_test = y_full[5000:]

# Setup input and label placeholders
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape = (None, n - 1), name="X")
    y = tf.placeholder(tf.int32 , shape = (None), name="y")

# Add the hidden layers.
h1 = tf.layers.dense(X, 32, activation=tf.nn.sigmoid, name="h1")
h2 = tf.layers.dense(h1, 32, activation=tf.nn.sigmoid, name="h2")
h3 = tf.layers.dense(h2, 48, activation=tf.nn.sigmoid, name="h3")
h4 = tf.layers.dense(h3, 56, activation=tf.nn.sigmoid, name="h4")
h5 = tf.layers.dense(h4, 32, activation=tf.nn.sigmoid, name="h5")


KEEP_RATE = tf.Variable(.5)
# Add some droupout to each layer to prevent overtraining.
tf.nn.dropout(h1, KEEP_RATE)
tf.nn.dropout(h2, KEEP_RATE)
tf.nn.dropout(h3, KEEP_RATE)
tf.nn.dropout(h4, KEEP_RATE)
tf.nn.dropout(h5, KEEP_RATE)

# Add output
with tf.name_scope("output"):
    logits = tf.layers.dense(h5, max(y_full)+1, name="logits")

# Define the training operations
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

# Add evaluation operations
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, TOP_K_LOGITS)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    conf = tf.confusion_matrix(y, tf.argmax(logits, axis=1))

# Initializer and saver. Saver won't actually be used yet,
# but it's there if I decide to open up TensorBoard
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

for keep_rate in np.arange(0.5, 0.6, 0.1):
    KEEP_RATE.assign(keep_rate)
    with tf.Session() as sess:
        init.run()
        for epoch in range(EPOCHS):
            for i in range(len(x_train) // BATCH_SIZE):

                # Get batch for this iteration
                x_batch = x_train[(BATCH_SIZE*i) : (BATCH_SIZE*(i+1))]
                y_batch = y_train[(BATCH_SIZE*i) : (BATCH_SIZE*(i+1))]

                # Train with this batch
                sess.run(training_op, feed_dict={X: x_batch, y: y_batch})

            # Evaluate on the test set every epoch
            #print("Epoch %d accuracy test:"%epoch, accuracy.eval(feed_dict={X: x_test, y: y_test}),
            #      "train:", accuracy.eval(feed_dict={X: x_train, y: y_train}))

        # Turn off dropout for testing
        KEEP_RATE.assign(1.0)

        # Evaluate on the full set and save to a confusion matrix after training is complete
        #print("Full dataset accuracy: ", accuracy.eval(feed_dict={X: x_full, y: y_full}))
        print(accuracy.eval(feed_dict={X: x_test, y: y_test}))
        print("Specificity for %f keep rate:"%keep_rate, accuracy.eval(feed_dict={X: x_train, y: y_train})
              - accuracy.eval(feed_dict={X: x_test, y: y_test}))
        plot_confusion_matrix(conf.eval(feed_dict={X: x_full, y: y_full}), range(0, max(y_test) + 1), normalize=True)

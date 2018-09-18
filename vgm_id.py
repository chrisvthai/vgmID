import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import numpy as np
import random

from spectro_helper import spectrogram, prepare_data, read_wav_array, read_single_wav
from sklearn import preprocessing

def main(_):

    x = tf.placeholder(tf.float32, shape=[None, 128, 256, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 7])

    initializer = tf.contrib.layers.xavier_initializer()

    # [batch_size, 128, 256, 1]
    conv_1 = tf.layers.conv2d(
          inputs=x,
          filters=32,
          kernel_size=5,
          padding='same',
          activation=tf.nn.relu,
          kernel_initializer=initializer
          )

    # [batch_size, 128, 256, 32]

    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2,2], strides=2)
    # [batch_size, 64, 128, 32]

    conv_2 = tf.layers.conv2d(
          inputs=pool_1,
          filters=64,
          kernel_size=5,
          padding='same',
          activation=tf.nn.relu,
          kernel_initializer=initializer
          )
    # [batch_size, 64, 128, 64]

    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2,2], strides=2)
    # [batch_size, 32, 64, 64]

    pool_2_flat = tf.reshape(pool_2, [-1, 32*64*64])

    dense_1 = tf.layers.dense(inputs=pool_2_flat, units=2048, activation=tf.nn.relu, kernel_initializer=initializer)
    dropout = tf.layers.dropout(inputs=dense_1, rate=0.4)
    y_conv  = tf.layers.dense(inputs=dropout, units=7, kernel_initializer=initializer) 

    # Train step data
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))

    learn_rate = 1e-4
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # To the actual training loop!
    saver = tf.train.Saver()
    path = "./vgm-id-model/"
    if not os.path.exists(path):
      os.makedirs(path)

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, path + 'test-model')
        file_list, y_wav_label = prepare_data(FLAGS.wav_dir)
        le = preprocessing.LabelEncoder()
        # labels = ['Adventure', 'Casual', 'Fighting', 'Horror', 'Platformer', 'RPG', 'Shooter', 'Strategy']
        y_one_hot = tf.one_hot(le.fit_transform(y_wav_label),depth=7)
        y_whole = sess.run(y_one_hot)

        if FLAGS.train:
            # Get all possible one-hot encodings for the different categories first
            
            batch_size = 50

            for i in range(10000):
            	# Get samples of filenames and their labels. Filenames first, then one-hot labels
                indices = random.sample(range(len(file_list)), batch_size)
                #print(indices)
                filename_indices = []
                for j in indices:
                    filename_indices.append(file_list[j])  
                label_batch = tf.stack(tf.gather(y_whole, indices))

                x_feed = sess.run(read_wav_array(filename_indices))
                y_feed = sess.run(label_batch)

                #Training here
                if i % 3 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: x_feed , y_: y_feed})
                    print('step %d, training accuracy %g' % (i, train_accuracy))

                train_step.run(feed_dict={x: x_feed , y_: y_feed})
                save_path = saver.save(sess, path + 'test-model')

        elif FLAGS.predict_wav != "":
        	predicted = tf.argmax(y_conv, 1)
        	x_single_wav = sess.run(read_single_wav(FLAGS.predict_wav))
        	print("Is", FLAGS.predict_wav, "from a", le.inverse_transform(sess.run(predicted, feed_dict={x: x_single_wav}))[0], "game?")

        elif FLAGS.test:
            batch_size = 50
            batch_files = []
            batch_indices = []
            indices_all = list(range(len(file_list)))

            for i in range(0, len(file_list), batch_size):
                batch_files.append(file_list[i:i+batch_size])
                batch_indices.append(indices_all[i:i+batch_size])

            batch_accuracies = []
            for files, indices in zip(batch_files, batch_indices):
                x_feed = sess.run(read_wav_array(files))

                labels = tf.stack(tf.gather(y_whole, indices))
                y_feed = sess.run(labels)

                test_accuracy = accuracy.eval(feed_dict={x: x_feed, y_: y_feed})
                batch_accuracies.append(test_accuracy)

            print(batch_accuracies)
            avg_accuracy = sum(batch_accuracies) / len(batch_accuracies)
            print("Average test accuracy on the training data is: %g" % avg_accuracy)

           # x_feed = sess.run(read_wav_array(file_list))
           # y_feed = y_whole

            #test_accuracy = accuracy.eval(feed_dict={x: x_feed, y_: y_feed})
            #print('Test accuracy: %d' % (train_accuracy))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
        '--wav_dir',
        type=str,
        default='songs',
        help='Path to folders of labeled audio files.'
  )
  parser.add_argument(
        '--predict_wav',
        type=str,
        default="",
        help='Try to predict an unknown wav file'
    )
  parser.add_argument(
        '--train',
        help='Train the network',
        action='store_true'
        )
  parser.add_argument(
  	    '--test',
  	    help='Get average test accuracy across training data',
  	    action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

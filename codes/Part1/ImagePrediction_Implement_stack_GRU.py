#Import packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.ops import rnn, rnn_cell
import random
import os
import numpy as np

mnist = input_data.read_data_sets("../../data",one_hot=True)
#Get the Data
train_image = mnist.train.images
train_label = mnist.train.labels
test_image = mnist.test.images
test_label = mnist.test.labels


#Data pre-process
def binarization(images,threshold):
    return (threshold<images).astype('float32')

train_image_input = binarization(train_image, 0.1)
test_image_input = binarization(test_image, 0.1)
print('Data pre-process finished!')

#Classification Classes
n_classes = 10

#Defining the chuncks *how many will be processed at each time*
chunck_size = 1
n_chunks = 784
rnn_size = 32
restore_dict = '../../Model/model_Heiarchy_GRU/ImagePrediction.checkpoint'

#Get the x input as float and reshape input
x = tf.placeholder("float",[None,n_chunks,chunck_size])  #Batch_size * n_chunk * chunk_size
y = tf.placeholder("float",[None,n_classes])             #Batch_size * n_classes

def Recurrent_neural_network(x):
    #Defining the sigle layer RNN            
    Data_Processed = tf.transpose(x,[1,0,2])      #n_chunk * batch_size * chunk_size
    Data_Processed = tf.reshape(Data_Processed,[-1,chunck_size])    #(n_chunk*batch_size) * chunk_size
    Data_Processed = tf.split(Data_Processed, num_or_size_splits=n_chunks, axis=0)  # n_chunk {batch_size*chunk_size}
    # lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
    # All_Outputs, All_States = tf.contrib.rnn.static_rnn(lstm_cell,Data_Processed,dtype=tf.float32)
    GRU_cell = tf.contrib.rnn.GRUCell(rnn_size)
    Heiarchy_RNN = tf.contrib.rnn.MultiRNNCell([GRU_cell] * 3)
    All_Outputs, All_States =tf.contrib.rnn.static_rnn(Heiarchy_RNN,Data_Processed,dtype=tf.float32)
        
    # Post-RNN process
    Process_output = tf.contrib.layers.linear(All_Outputs[-1], 100)    # [batch_size x 100]
    Process_output = tf.nn.relu(Process_output)                        # [batch_size x 100 RELU processed]
    final_output = tf.contrib.layers.linear(Process_output, n_classes)  # [batch_size x 10]

    return final_output


def Implement_neural_network(x):
    prediction = Recurrent_neural_network(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    with tf.Session() as sess:
        # load the model
        saver = tf.train.Saver()
        saver.restore(sess, restore_dict)
        # run the session
        #test_feed_dict = {x: test_image_input.reshape(-1, n_chunks, chunck_size), y: test_label}
        #predicted = sess.run(prediction, feed_dict=test_feed_dict)
        train_dict = {x:train_image_input.reshape(-1,n_chunks,chunck_size), y:train_label}
        test_dict = {x:test_image_input.reshape(-1,n_chunks,chunck_size), y:test_label}
        correction_check = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correction_check, 'float'))
        print('The final Train Accuracy is',accuracy.eval(train_dict))
        print('The final Test Accuracy is', accuracy.eval(test_dict))
        print('The Train loss of this kind of GRU RNN is', loss.eval(train_dict))
        print('The Test loss of this kind of GRU RNN is',loss.eval(test_dict))

#Implement
Implement_neural_network(x)

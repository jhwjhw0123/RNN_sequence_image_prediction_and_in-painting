#Import tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import os

#Select Different Modes
Pixel_Prediction_Flag = 32   #64,128

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
#print(train_image[1])
#print(train_image_input[1])

#Classification Classes
n_classes = 2
#Trainning size
batch_size = 512
#Defining training epochs (iterations)
nEpochs = 100

#Defining the chuncks *how many will be processed at each time*
chunck_size = 1
n_chunks = 784
if Pixel_Prediction_Flag == 32:
    rnn_size = 32
    restore_dict = '../../Model/model_LSTM_32/PixelPrediction.checkpoint'
elif Pixel_Prediction_Flag == 64:
    rnn_size = 64
    restore_dict = '../../Model/model_LSTM_64/PixelPrediction.checkpoint'
elif Pixel_Prediction_Flag == 128:
    rnn_size = 128
    restore_dict = '../../Model/model_LSTM_128/PixelPrediction.checkpoint'
#Get the x input as float and reshape input
x = tf.placeholder("float",[None,n_chunks,chunck_size])  #Batch_size * n_chunk * chunk_size
y = tf.placeholder("float",[None,chunck_size])    #[Batch_size * n_chunks] * 1 (not one_hot encoding)

def Recurrent_neural_network(x):
    #Defining the sigle layer RNN

    current_batch_size = tf.shape(x)[0]
    Rnn_to_pixel_layer = {'weight':tf.Variable(tf.random_normal([rnn_size,1])),\
                     'bias':tf.Variable(tf.random_normal([1]))}

    Data_Processed = tf.transpose(x,[1,0,2])      #n_chunk * batch_size * chunk_size
    Data_Processed = tf.reshape(Data_Processed,[-1,chunck_size])    #(n_chunk*batch_size) * chunk_size
    Data_Processed = tf.split(Data_Processed,num_or_size_splits=n_chunks,axis=0)     #n_chunk * [batch_size*chunk_size]

    lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
    #Heiarchy_RNN = tf.contrib.rnn.MultiRNNCell([lstm_cell]*3)
    All_Outputs, All_States = tf.contrib.rnn.static_rnn(lstm_cell,Data_Processed,dtype=tf.float32)

    # Post-RNN process
    Pixel_output = []
    for cPixel in range(len(All_Outputs)-1):
        This_pixel = tf.add(tf.matmul(All_Outputs[cPixel],Rnn_to_pixel_layer['weight']),Rnn_to_pixel_layer['bias'])    #batch_size * 1
        #This_pixel = tf.contrib.layers.linear(All_Outputs[cPixel],chunck_size)
        Pixel_output.append(This_pixel)
    #Pixel_output: n_chunks * [batch_size*1]
    Concat_Pixel_output = tf.concat(Pixel_output,axis=0)       #n_chunks-1 * batch_size*1
    
    final_output = tf.reshape(Concat_Pixel_output,[(n_chunks-1) * current_batch_size,-1])

    return final_output

def Implement_neural_network(x):
    prediction = Recurrent_neural_network(x)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    prediction_eval = tf.nn.sigmoid(prediction)

    #Pre-process test Data
    test_x = test_image_input.reshape([-1,n_chunks,chunck_size])
    test_y = test_x.transpose([1,0,2])    # n_chunk * m_amount * chunck_size
    test_y = test_y[1:]
    test_y = test_y.reshape([-1, chunck_size])

    #Pre_process Train Data
    train_x = train_image_input.reshape([-1,n_chunks,chunck_size])
    train_y = train_x.transpose([1,0,2])    # n_chunk * m_amount * chunck_size
    train_y = train_y[1:]
    train_y = train_y.reshape([-1, chunck_size])

    with tf.Session() as sess:
        # load the model
        saver = tf.train.Saver()
        saver.restore(sess, restore_dict)
        test_dict = {x:test_x, y:test_y}
        train_dict = {x:train_x,y:train_y}
        correction_check = tf.equal(tf.round(prediction_eval),y)
        accuracy = tf.reduce_mean(tf.cast(correction_check, 'float'))
        print('The final Training Accuracy is', accuracy.eval(train_dict))
        print('The final Test Accuracy is',accuracy.eval(test_dict))
        print('The Training cross-entropy loss',loss.eval(train_dict))
        print('The Test cross-entropy loss',loss.eval(test_dict))


#Implement above functions
Implement_neural_network(x)

    

    

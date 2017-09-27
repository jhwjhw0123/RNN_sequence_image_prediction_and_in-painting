#Import packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import os
import numpy as np

mnist = input_data.read_data_sets("../../Data",one_hot=True)
#Get the Data
train_image = mnist.train.images
train_label = mnist.train.labels
test_image = mnist.test.images
test_label = mnist.test.labels

#Get the plotting data dictionary
#if not os.path.exists('./Plot/Data/Question1/'):
#	os.mkdir('./Plot/Data/Question1/')

# save the model params to the hard drive
def save_model(session):
    if not os.path.exists('../../Model/model_128_tf_1_0/'):
        os.mkdir('../../Model/model_128_tf_1_0/')
    saver = tf.train.Saver()
    saver.save(session, '../../Model/model_128_tf_1_0/ImagePrediction.checkpoint')

#Data pre-process
def binarization(images,threshold):
    return (threshold<images).astype('float32')

train_image_input = binarization(train_image, 0.1)
test_image_input = binarization(test_image, 0.1)
print('Data pre-process finished!')
#print(train_image[1])
#print(train_image_input[1])

#Classification Classes
n_classes = 10
#Trainning size
batch_size = 512
#Defining maximum training epochs (iterations)
nEpochs = 1000

#Defining the chuncks *how many will be processed at each time*
chunck_size = 1
n_chunks = 784
rnn_size = 128

#Get the x input as float and reshape input
x = tf.placeholder("float",[None,n_chunks,chunck_size])  #Batch_size * n_chunk * chunk_size
y = tf.placeholder("float",[None,n_classes])             #Batch_size * n_classes

def Recurrent_neural_network(x):
    #Data shape adjust to feed to RNN		 
    Data_Processed = tf.transpose(x,[1,0,2])      #n_chunk * batch_size * chunk_size
    Data_Processed = tf.reshape(Data_Processed,[-1,chunck_size])    #(n_chunk*batch_size) * chunk_size
    Data_Processed = tf.split(Data_Processed, num_or_size_splits=n_chunks, axis=0)  # n_chunk {batch_size*chunk_size}
    lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
    #Heiarchy_RNN = tf.contrib.rnn.MultiRNNCell([lstm_cell]*3)
    All_Outputs, All_States = tf.contrib.rnn.static_rnn(lstm_cell,Data_Processed,dtype=tf.float32)
    #GRU_cell = tf.contrib.GRUCell(rnn_size)
    #All_Outputs, All_States =tf.contrib.static_rnn(GRU_cell,Data_Processed,dtype=tf.float32)
		
    # Post-RNN process
    Process_output = tf.contrib.layers.linear(All_Outputs[-1], 100)    # [batch_size x 100]
    Process_output = tf.nn.relu(Process_output)                        # [batch_size x 100 RELU processed]
    final_output = tf.contrib.layers.linear(Process_output, n_classes)  # [batch_size x 10]

    return final_output

def train_neural_network(x):
    prediction = Recurrent_neural_network(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    Accuracy_train_collection = []
    Accuracy_test_collection = []

    #Defining the optisimiser
    #learning rate = 0.001
    OriginalOptimiser = tf.train.AdamOptimizer(0.001)
    #gradient clipping
    grds = OriginalOptimiser.compute_gradients(loss)
    capped_grds = [(tf.clip_by_value(grdvalue, -7., 7.), var) for grdvalue, var in grds]
    optimiser = OriginalOptimiser.apply_gradients(capped_grds)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prev_test_accuracy = 0
        DropdownTime = 0
        #Training Session
        for cEpoch in range(nEpochs):
            current_Epoch_Loss = 0
            n = train_image_input.shape[0]
            random_index = random.sample(range(n), n)
            #for each epoch we need times to perform stochastic gradient descent 
            for i in range(n//batch_size):
                current_x = train_image_input[random_index[i * batch_size: (i + 1) * batch_size]]
                current_y = train_label[random_index[i * batch_size: (i + 1) * batch_size]]
                current_x = current_x.reshape([batch_size,n_chunks,chunck_size])
                _,currentloss = sess.run([optimiser,loss],feed_dict = {x:current_x,y:current_y})
                current_Epoch_Loss += currentloss
            print(cEpoch+1,'iteration has been completed and the loss of this epoch is',current_Epoch_Loss)
            correction_check = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))     #Batch_size * 1
            accuracy = tf.reduce_mean(tf.cast(correction_check,'float'))
            train_accuracy = accuracy.eval({x:train_image_input.reshape(-1,n_chunks,chunck_size),y:train_label})
            test_accuracy = accuracy.eval({x:test_image_input.reshape(-1,n_chunks,chunck_size),y:test_label})
            Accuracy_train_collection.append(train_accuracy)
            Accuracy_test_collection.append(test_accuracy)
            print('The Train Accuracy of the',cEpoch+1,'itetation is',train_accuracy)
            print('The Test Accuracy of the',cEpoch+1,'itetation is',test_accuracy)
            if test_accuracy>=0.9:
            	np.save('../Plot/Data/Question1/TrainAccuracy_128_tf_1_0',np.asarray(Accuracy_train_collection))
            	np.save('../Plot/Data/Question1/TestAccuracy_128_tf_1_0',np.asarray(Accuracy_test_collection))
            	save_model(sess)
            	if test_accuracy> 0.95 and test_accuracy<prev_test_accuracy:
                    DropdownTime = DropdownTime + 1
                    if DropdownTime>=5:
                        break
            elif cEpoch == nEpochs-1:
            	np.save('../Plot/Data/Question1/TrainAccuracy_128_tf_1_0',np.asarray(Accuracy_train_collection))
            	np.save('../Plot/Data/Question1/TestAccuracy_128_tf_1_0',np.asarray(Accuracy_test_collection))
            	if test_accuracy>=0.9:
            		save_model(sess)
            prev_test_accuracy = test_accuracy

        correction_check = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correction_check, 'float'))
        print('The final Test Accuracy is', accuracy.eval({x:test_image_input.reshape(-1,n_chunks,chunck_size), y:test_label}))
        
#Implement above functions
train_neural_network(x)
    
    

    

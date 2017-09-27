#Import tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import os

mnist = input_data.read_data_sets("../../data",one_hot=True)
#Get the Data
train_image = mnist.train.images
train_label = mnist.train.labels
test_image = mnist.test.images
test_label = mnist.test.labels

# save the model params to the hard drive
def save_model(session):
    if not os.path.exists('../../Model/model_LSTM_Heiarchy/'):
        os.mkdir('../../Model/model_LSTM_Heiarchy/')
    saver = tf.train.Saver()
    saver.save(session, '../../Model/model_LSTM_Heiarchy/PixelPrediction.checkpoint')

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
rnn_size = 32

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
    Heiarchy_RNN = tf.contrib.rnn.MultiRNNCell([lstm_cell]*3)
    All_Outputs, All_States = tf.contrib.rnn.static_rnn(Heiarchy_RNN,Data_Processed,dtype=tf.float32)

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

def train_neural_network(x):
    prediction = Recurrent_neural_network(x)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    prediction_eval = tf.nn.sigmoid(prediction)

    #Defining the optisimiser
    #learning rate defaulte = 0.001
    optimiser = tf.train.AdamOptimizer().minimize(loss)

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
        sess.run(tf.global_variables_initializer())


        Accuracy_train_collection = []
        Accuracy_test_collection = []
        prev_test_accuracy = 0

        #Training Session
        for cEpoch in range(nEpochs):
            current_Epoch_Loss = 0
            n = train_image_input.shape[0]
            random_index = random.sample(range(n), n)
            #for each epoch we need times to perform stochastic gradient descent
            for i in range(n//batch_size):
                current_x = train_image_input[random_index[i * batch_size: (i + 1) * batch_size]]
                current_x = current_x.reshape([batch_size,n_chunks,chunck_size])
                current_y = current_x.transpose([1,0,2])    # n_chunk * m_amount * chunck_size
                current_y = current_y[1:]
                current_y = current_y.reshape([(n_chunks-1)*batch_size,chunck_size])
                _,currentloss = sess.run([optimiser,loss],feed_dict = {x:current_x,y:current_y})
                current_Epoch_Loss += currentloss
            print(cEpoch+1,'iteration has been completed and the loss of this epoch is',current_Epoch_Loss)
            correction_check = tf.equal(tf.round(prediction_eval),y)           #Not one_hot encoding
            #prediction: [n_chunks * batch_size] * 1
            #y:[n_chunks * batch_size] * 1
            accuracy = tf.reduce_mean(tf.cast(correction_check,'float'))
            train_accuracy = accuracy.eval({x:train_x,y:train_y})
            test_accuracy = accuracy.eval({x:test_x,y:test_y})
            Accuracy_train_collection.append(train_accuracy)
            Accuracy_test_collection.append(test_accuracy)
            print('The Train Accuracy of the',cEpoch+1,'itetation is',train_accuracy)
            print('The Test Accuracy of the',cEpoch+1,'itetation is',test_accuracy)
            if test_accuracy> 0.95 and test_accuracy<prev_test_accuracy:
                np.save('../Plot/Data/Question2/TrainAccuracy_Heiarchy',np.asarray(Accuracy_train_collection))
                np.save('../Plot/Data/Question2/TestAccuracy_Heiarchy',np.asarray(Accuracy_test_collection))
                save_model(sess)
                break
            elif cEpoch == nEpochs-1:
                np.save('../Plot/Data/Question2/TrainAccuracy_Heiarchy',np.asarray(Accuracy_train_collection))
                np.save('../Plot/Data/Question2/TestAccuracy_Heiarchy',np.asarray(Accuracy_test_collection))
                save_model(sess)
            prev_test_accuracy = test_accuracy

        correction_check = tf.equal(tf.round(prediction_eval),y)
        accuracy = tf.reduce_mean(tf.cast(correction_check, 'float'))
        print('The final Accuracy is', accuracy.eval({x:test_x, y:test_y}))


#Implement above functions
train_neural_network(x)

    

    

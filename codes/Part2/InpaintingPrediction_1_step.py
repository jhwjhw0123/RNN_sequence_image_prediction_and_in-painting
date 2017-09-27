#Import tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import os
#Import the Data
CorruptedImage = np.load('../CorruptData/CorrputLast300.npy')
GroundTruth = np.load('../CorruptData/GroundTrust.npy')
ImageAmount = CorruptedImage.shape[0]
ImageRange = CorruptedImage.shape[1]

#Classification Classes
n_classes = 2
#Trainning size
batch_size = 512
#Defining training epochs (iterations)
nEpochs = 100
Prediction_Start_ind = 484

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
    Concat_Pixel_output = tf.concat(Pixel_output,axis=0)       #n_chunks * batch_size*1
    
    final_output = tf.reshape(Concat_Pixel_output,[(n_chunks-1) * current_batch_size,-1])

    return final_output

def InPainting(x):
    prediction = Recurrent_neural_network(x)
    prediction_eval = tf.nn.sigmoid(prediction)     #n_chunck * batch_size * 1
    # Calculate prediction result
    predict_result = tf.round(prediction_eval)  # n_chunck * batch_size * 1
    # Split it for indexing
    Pixelwise_prediction = tf.split(predict_result, num_or_size_splits=ImageRange-1, axis=0)  # n_chunck-1 * [batch_size*1]

    #Pre-process test Data
    test_x = CorruptedImage.reshape([-1,n_chunks,chunck_size])
    true_x = GroundTruth.reshape([-1,n_chunks,chunck_size])
    test_y = true_x.transpose([1,0,2])    # n_chunk * m_amount * chunck_size
    test_y = test_y.reshape([-1, chunck_size])
    #Initialize Dynamic Test Pixel Data
    input_pixel_x = test_x
    input_pixel_y = test_y
    previous_iamge = CorruptedImage
    #Get the ground truth list to determine the accuracy
    GroundTruthList = np.split(GroundTruth,ImageRange,axis=1)   #784 * [100]
    #Get the real sequence and define the sequence probability
    Real_sequence = GroundTruthList[Prediction_Start_ind:Prediction_Start_ind+1]     #for 1 pixel prediction
    Inpainting_sequence_prob = []
    Groundtruth_sequence_prob = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, '../../Model/model_LSTM_Heiarchy/PixelPrediction.checkpoint')
        for i in range(1):
            pixel_index = i + Prediction_Start_ind - 1
            InPainting_dict = {x:input_pixel_x,y:input_pixel_y}
            Ground_truth_dict = {x:true_x,y:test_y}
            #**********************Information from RNN**************************
            # Calculate the value of the current pixel
            Predicted_pixel_value = Pixelwise_prediction[pixel_index].eval(InPainting_dict)  # 100 * 1
            Inpainting_pixel_value = np.reshape(Predicted_pixel_value,[ImageAmount])       # 100 scalar
            # Split the previous image
            flat_PreviousImage = np.split(previous_iamge, ImageRange, axis=1)  # 784*[100*1]
            Ground_truth_pixel_value = np.reshape(GroundTruthList[pixel_index],[ImageAmount])   #100 scalar
            # replace the current pixel
            flat_PreviousImage[pixel_index+1] = Predicted_pixel_value  # 784*[100*1]
            #*******************Probability sample and calculation**************
            #Calculate the pixel probability
            Inpainting_prob = prediction_eval.eval(InPainting_dict)
            Inpainting_prob_list = np.split(Inpainting_prob,ImageRange-1,axis=0)        #783 * [100*1]
            Inpainting_pixel_one_prob = Inpainting_prob_list[pixel_index]              #100 * 1
            Inpainting_pixel_prob = np.reshape(Inpainting_pixel_one_prob,[ImageAmount])   #100 scalar
            Ground_truth_prob = prediction_eval.eval(Ground_truth_dict)
            Ground_truth_prob_list = np.split(Ground_truth_prob, ImageRange-1, axis=0)  # 783 * [100*1]
            Ground_truth_pixel_one_prob = Ground_truth_prob_list[pixel_index]  # 100 * 1
            Groundtruth_pixel_prob = np.reshape(Ground_truth_pixel_one_prob, [ImageAmount])  # 100 scalar
            # This_pixel_one_prob = np.repeat(This_pixel_one_prob,10, axis=0)
            # This_pixel_one_prob_sample = np.reshape(This_pixel_one_prob,[ImageAmount,10,1])
            # #Ceate a sampling matrix
            # Sampling_matrix = np.random.uniform(low=0.0, high=1.0, size=This_pixel_one_prob_sample.shape)
            # #Get the sample of this step
            # This_pixel_sample = This_pixel_one_prob_sample - Sampling_matrix
            # This_pixel_sample[This_pixel_sample>=0] = 1
            # This_pixel_sample[This_pixel_sample<0] = 0
            # for i in range(Inpainting_pixel_one_prob.shape[0]):
            #     if Inpainting_pixel_value[i] == 0:
            #         Inpainting_pixel_prob[i] = 1 - Inpainting_pixel_prob[i]
            #     if Ground_truth_pixel_value[i] == 0:
            #         Groundtruth_pixel_prob[i] = 1 - Groundtruth_pixel_prob[i]
            #***************************Update*************************************
            # concatenate the image again
            new_image = np.stack(flat_PreviousImage, axis=0)  # 784*100*1
            previous_iamge = np.reshape(new_image, [ImageAmount, ImageRange])
            # Update the input feed dictionary
            input_pixel_x = new_image.transpose([1, 0, 2])
            #****************************Evaluate*****************************
            #calculate cross-entropy
            Inpainting_cross_entropy = -(np.multiply(Inpainting_pixel_value, np.log(Inpainting_pixel_prob)) + np.multiply((1 - Inpainting_pixel_value), np.log(1 - Inpainting_pixel_prob)))
            Ground_truth_cross_entropy = -(np.multiply(Ground_truth_pixel_value, np.log(Groundtruth_pixel_prob)) + np.multiply((1 - Ground_truth_pixel_value), np.log(1 - Groundtruth_pixel_prob)))
            Full_batch_output_this_pixel = np.reshape(flat_PreviousImage[pixel_index],[ImageAmount])
            Full_batch_ground_truth_this_pixel = np.reshape(GroundTruthList[pixel_index],[ImageAmount])
            correction_check = np.equal(Full_batch_output_this_pixel,Full_batch_ground_truth_this_pixel)
            accuracy = (np.sum(correction_check)*1.0)/(correction_check.shape[0]*1.0)
            print('The prediction Accuracy is', '%.3f'%round(accuracy,3))
            print('The In-painting cross-entropy of the 1-step painting is:',Inpainting_cross_entropy)
            print('The Ground_truth cross-entropy of the 1-step painting is:', Ground_truth_cross_entropy)
            print(Inpainting_cross_entropy-Ground_truth_cross_entropy)
            np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Holistic_Inpainting_entropy_1_step',Inpainting_cross_entropy)
            np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Holistic_Ground_Truth_entropy_1_step',Ground_truth_cross_entropy)
            np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/accuracy_10',accuracy)
            np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Reconstructed_Image_10',input_pixel_x)


#Implement above functions
InPainting(x)

    

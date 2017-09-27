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
Prediction_Length = 10

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
    prediction_eval = tf.nn.sigmoid(prediction)     #(n_chunck-1) * batch_size * 1
    # Calculate prediction result
    predict_result = tf.round(prediction_eval)  # (n_chunck-1) * batch_size * 1
    # Split it for indexing
    Pixelwise_prediction = tf.split(predict_result, num_or_size_splits=ImageRange-1, axis=0)  # n_chunck * [batch_size*1]

    #Pre-process test Data
    Sample_CorruptedImage = np.repeat(CorruptedImage, 10, axis=0)
    Sample_GroundTruth = np.repeat(GroundTruth, 10, axis=0)
    test_x = Sample_CorruptedImage.reshape([-1,n_chunks,chunck_size])
    true_x = GroundTruth.reshape([-1,n_chunks,chunck_size])
    test_y = true_x.transpose([1,0,2])    # n_chunk * m_amount * chunck_size
    test_y = test_y.reshape([-1, chunck_size])
    true_x_inpaint = Sample_GroundTruth.reshape([-1, n_chunks, chunck_size])
    test_y_inpaint = true_x_inpaint.transpose([1, 0, 2])  # n_chunk * (10*m_amount) * chunck_size
    test_y_inpaint = test_y_inpaint.reshape([-1, chunck_size])
    #Define the reconstruction array
    Image_reconstruction = CorruptedImage   #100*784
    #Initialize Dynamic Test Pixel Data for In Painting
    input_pixel_x = test_x
    input_pixel_y = test_y_inpaint
    previous_iamge = Sample_CorruptedImage    #1000 Images
    #Get the ground truth list to determine the accuracy
    GroundTruthList = np.split(GroundTruth,ImageRange,axis=1)   #784 * [100]
    #define the sequence entropy
    Inpainting_sequence_entropy = []
    Groundtruth_sequence_entropy = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, '../../Model/model_LSTM_Heiarchy/PixelPrediction.checkpoint')
        for i in range(Prediction_Length):
            pixel_index = i + Prediction_Start_ind - 1
            InPainting_dict = {x:input_pixel_x,y:input_pixel_y}
            #print(np.shape(input_pixel_x))
            Ground_truth_dict = {x:true_x,y:test_y}
            #**********************Information from RNN**************************
            # Calculate the value of the current pixel
            Predicted_pixel_value = Pixelwise_prediction[pixel_index].eval(InPainting_dict)  # 1000 * 1
            Inpainting_pixel_value = Predicted_pixel_value[0::10]    # 100 *1 array
            #Deterministic Image Recostruction
            flat_Image_reconstruction = np.split(Image_reconstruction,ImageRange,axis=1)   #784 * [100]
            flat_Image_reconstruction[pixel_index+1] = Inpainting_pixel_value
            Image_reconstruction = np.reshape(np.stack(flat_Image_reconstruction, axis=1),[ImageAmount,ImageRange])  #100*784
            # Split the previous image
            #flat_PreviousImage = np.array_split(input_pixel_x, ImageRange, axis=1)  # 784*[1000*1]
            flat_PreviousImage = np.transpose(input_pixel_x,[1,0,2])
            #print(np.shape(flat_PreviousImage))
            Ground_truth_pixel_value = np.reshape(GroundTruthList[pixel_index],[ImageAmount])   #100 array
            # *******************Probability sample and calculation**************
            # Calculate the pixel probability
            Inpainting_prob = prediction_eval.eval(InPainting_dict)
            Inpainting_prob = np.reshape(Inpainting_prob,[ImageRange-1,10*ImageAmount,1])
            Inpainting_prob_list = np.split(Inpainting_prob, ImageRange-1, axis=0)  # 783 predictions * [1000*1]
            Inpainting_pixel_one_prob = Inpainting_prob_list[pixel_index]  # 1000 * 1
            Inpainting_pixel_prob = np.reshape(Inpainting_pixel_one_prob, [10*ImageAmount])  # 1000 array
            Ground_truth_prob = prediction_eval.eval(Ground_truth_dict)
            Ground_truth_prob_list = np.split(Ground_truth_prob, ImageRange-1, axis=0)  # 783 * [100*1]
            Ground_truth_pixel_one_prob = Ground_truth_prob_list[pixel_index]  # 100 * 1
            Groundtruth_pixel_prob = np.reshape(Ground_truth_pixel_one_prob, [ImageAmount])  # 100 array
            Predicted_gound_truth_pixel_value = Pixelwise_prediction[pixel_index].eval(Ground_truth_dict)  # 100* 1
            #print(Inpainting_pixel_value[0])
            #print(Predicted_gound_truth_pixel_value[0])
            #print(true_x[0][pixel_index+1])
            # 10 sample reshape
            # Ceate a sampling matrix
            Sampling_matrix = np.random.uniform(low=0.0, high=1.0, size=Inpainting_pixel_prob.shape)
            # Get the sample of this step
            InPainting_pixel_sample = Inpainting_pixel_prob - Sampling_matrix  # 100 * 10 * 1
            InPainting_pixel_sample[InPainting_pixel_sample >= 0] = 1
            InPainting_pixel_sample[InPainting_pixel_sample < 0] = 0
            #print(InPainting_pixel_sample.shape)
            # replace the current pixel
            #print(np.shape(InPainting_pixel_sample))
            #print('\n')
            flat_PreviousImage[pixel_index+1] = np.reshape(InPainting_pixel_sample,[10*ImageAmount,1])  # 784*[1000*1]
            # for i in range(Inpainting_pixel_one_prob.shape[0]):        #1000 iteration
            #     if InPainting_pixel_sample[i] == 0:
            #         Inpainting_pixel_prob[i] = 1 - Inpainting_pixel_prob[i]
            # for i in range(Groundtruth_pixel_prob.shape[0]):           #100 iteration
            #     if Ground_truth_pixel_value[i] == 0:
            #         Groundtruth_pixel_prob[i] = 1 - Groundtruth_pixel_prob[i]
            #Until now: Inpainting_pixel_prob: 1000 array
            #           Groundtruth_pixel_prob: 100 array
            #***************************Update*************************************
            # concatenate the image again
            #new_image = np.stack(flat_PreviousImage, axis=1)  # 1000*784*1
            new_image = np.transpose(flat_PreviousImage,[1,0,2])  # 1000*784*1
            #previous_iamge = np.reshape(new_image, [10*ImageAmount, ImageRange])
            # Update the input feed dictionary
            input_pixel_x = new_image
            #Store the probability
            #calculate cross-entropy
            Inpainting_cross_entropy = -(np.multiply(InPainting_pixel_sample, np.log(Inpainting_pixel_prob)) + np.multiply((1 - InPainting_pixel_sample), np.log(1 - Inpainting_pixel_prob)))   #1000 array
            Inpainting_cross_entropy_list = np.split(Inpainting_cross_entropy, ImageAmount, axis=0)
            Inpainting_cross_entropy_average = []
            for InPain_cross_entropy in Inpainting_cross_entropy_list:
                Inpainting_cross_entropy_average.append(np.mean(InPain_cross_entropy))
            Inpainting_cross_entropy_average = np.asarray(Inpainting_cross_entropy_average)
            Ground_truth_cross_entropy = -(np.multiply(Ground_truth_pixel_value, np.log(Groundtruth_pixel_prob)) + np.multiply((1 - Ground_truth_pixel_value), np.log(1 - Groundtruth_pixel_prob)))   #100 array
            #print(np.shape(Ground_truth_cross_entropy))
            #print(np.shape(Inpainting_cross_entropy_average))
            #store the entropy
            Inpainting_sequence_entropy.append(Inpainting_cross_entropy_average)
            Groundtruth_sequence_entropy.append(np.asarray(Ground_truth_cross_entropy))
        Holistic_Inpainting_entropy = np.sum(Inpainting_sequence_entropy,axis=0)
        Holistic_Ground_Truth_entropy = np.sum(Groundtruth_sequence_entropy,axis=0)
        print('The entropy of In-painting sequence is',Holistic_Inpainting_entropy)
        print('The entropy of Ground truth sequence is', Holistic_Ground_Truth_entropy)
        predict_sequence = np.split(Image_reconstruction,ImageRange,axis=1)[Prediction_Start_ind:Prediction_Start_ind+Prediction_Length]   #10 * 100
        true_sequence = np.split(GroundTruth,ImageRange,axis=1)[Prediction_Start_ind:Prediction_Start_ind+Prediction_Length]
        predict_sequence = np.reshape(predict_sequence,[Prediction_Length*ImageAmount,1])
        true_sequence = np.reshape(true_sequence,[Prediction_Length*ImageAmount,1])
        correction_check = np.equal(predict_sequence,true_sequence)
        #print(np.shape(predict_sequence))
        #print(np.shape(true_sequence))
        #correction_check = np.reshape(correction_check,)
        #print(np.shape(correction_check))
        #print(np.sum(correction_check))
        accuracy = (np.sum(correction_check)*1.0) / (correction_check.shape[0] * 1.0)
        print('The prediction Accuracy is', '%.3f'%round(accuracy,3))
        np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Holistic_Inpainting_entropy_10',Holistic_Inpainting_entropy)
        np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Holistic_Ground_Truth_entropy_10',Holistic_Ground_Truth_entropy)
        np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/accuracy_10',accuracy)
        np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Reconstructed_Image_10',Image_reconstruction)
        np.save('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Sampled_Image_10',new_image)

#Implement above functions
InPainting(x)

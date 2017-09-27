#Import tensorflow and necessary packages
import tensorflow as tf
import random
import numpy as np
import time
#Import the Data
OnePixelMissPackage = np.load('../CorruptData/2X2_pixels_inpainting.npy')
#Divide the Data into different parts
OnePixelMissInPainting = OnePixelMissPackage[0]
OnePixelMissGroundTruth = OnePixelMissPackage[1]
ImageAmount = OnePixelMissInPainting.shape[0]
ImageRange = OnePixelMissInPainting.shape[1]
# The amount the size for the two kinds of missing-pixel images are the same

#Classification Classes
n_classes = 2
#Trainning size
batch_size = 512

#Defining the chuncks *how many will be processed at each time*
chunck_size = 1
n_chunks = 784
rnn_size = 32    #Change Point1

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

def WindowPinxelInPainting(x):
    prediction = Recurrent_neural_network(x)
    prediction_eval = tf.nn.sigmoid(prediction)     #n_chunck * batch_size * 1
    # Calculate prediction result
    predict_result = tf.round(prediction_eval)  # n_chunck * batch_size * 1
    # Split it for indexing
    Imagewise_prediction = tf.reshape(predict_result,[n_chunks-1,16,1])  # n_chunck * 16* 1

    #Pre-process test Data
    test_x = OnePixelMissInPainting.reshape([-1,n_chunks,chunck_size])
    test_x_list = np.split(test_x,ImageAmount,axis=0)           #ImageAmount * [(1) * n_chunks * chunk_size]
    true_x = OnePixelMissGroundTruth.reshape([-1,n_chunks,chunck_size])
    true_x_list = np.split(true_x,ImageAmount,axis=0)           #ImageAmount * [(1) * n_chunks * chunk_size]
    #Get the Ground Truth list and the Inpainting Image list
    GroundTruthList = np.split(OnePixelMissGroundTruth,ImageAmount,axis=0)   #1000 * [1*784]
    InpaintingList = np.split(OnePixelMissInPainting,ImageAmount,axis=0)   #1000 * [1*784]
    

    #Get a list to collect the predicted images
    Predicted_Image_Collection = []
    Fill_in_digit_Collection = []
    InPainting_Cross_entropy_Collection = []
    GroundTruth_Cross_entropy_Collection = []
    Fill_in_index_collection = []
    ImageIndex = 0
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, '../../Model/model_LSTM_Heiarchy/PixelPrediction.checkpoint')    #Change_point2
        
        for image in InpaintingList:
            #image is 1*784 numpy array
            #Find the missing pixel index
            Missing_pixel_ind = np.where(image == -1)[1][0]
            Prediction_Length = ImageRange - Missing_pixel_ind
            input_pixel_x = np.reshape(test_x_list[ImageIndex],[1,n_chunks,chunck_size])
            input_pixel_x = np.repeat(input_pixel_x,16,axis=0)   # 16 * n_chunks * chunck_size
            #replace the correspoding pixel with all possible value
            for fourth_digit in range(2):
                for third_digit in range(2):
                    for second_digit in range(2):
                        for first_digit in range(2):
                            This_index = int(first_digit*(2^0) + second_digit*(2^1) + third_digit*(2^2) + fourth_digit*(2^3))
                            input_pixel_x[This_index][Missing_pixel_ind] = [first_digit]    #[0] = 0; [1] = 1
                            input_pixel_x[This_index][Missing_pixel_ind+1] = [second_digit]
                            input_pixel_x[This_index][Missing_pixel_ind+28] = [third_digit]
                            input_pixel_x[This_index][Missing_pixel_ind+29] = [fourth_digit]
                            #****************List of this rule of correponding value***********
                            #| 4 | 3 | 2 | 1 | value|
                            # 0    0   0   0     0
                            # 0    0   0   1     1
                            # 0    0   1   0     2
                            # ......  (Standard Binary Encoding-Decoding)
            input_pixel_true_y = np.reshape(true_x_list[ImageIndex],[1,n_chunks,chunck_size])
            input_pixel_true_y = np.repeat(input_pixel_true_y,16,axis=0)   # 16 * n_chunks * chunck_size
            input_pixel_y = input_pixel_true_y.transpose([1,0,2])    # n_chunk * m_amount(16) * chunck_size
            first_pixel = np.reshape(input_pixel_y[1], [1, 16, 1])  # 1 * amounts(16) *1
            first_pixel = first_pixel.transpose([1, 0, 2])
            input_pixel_y = input_pixel_y[1:]
            input_pixel_y = input_pixel_y.reshape([-1, chunck_size])
            #*********Notice that for this problem we don't need to change the feed dictionary step by step, thus for loop is nolonger needed**********
            MissingPixel_dict = {x:input_pixel_x,y:input_pixel_y}
            #Ground_truth_dict = {x:true_x,y:test_y}
            #**********************Information from RNN**************************
            # Calculate the value of the current pixel
            Predicted_Pixels = np.transpose(Imagewise_prediction.eval(MissingPixel_dict),[1,0,2])  # 16*[783 * 1]
            #Reconstruct the Image
            Predicted_Image = np.concatenate((first_pixel,Predicted_Pixels),axis=1)
            Predicted_Image_Collection.append(Predicted_Image)
            # *******************Probability and cross-entropy**************
            # Calculate the pixel probability
            Candidate_images_prob = prediction_eval.eval(MissingPixel_dict)   #(n_chunck-1 * 16(images)) * 1
            Candidate_images_prob = np.transpose(np.reshape(Candidate_images_prob,[n_chunks-1,16,1]),[1,0,2])        #16 * n_chunck-1 * 1
            GroundTruthInput = np.reshape(GroundTruthList[ImageIndex],[1,n_chunks,chunck_size])       #1 * n_chuncks * 1
            GroundTruthY = GroundTruthInput.transpose([1,0,2])
            GroundTruthY = GroundTruthY[1:]
            GroundTruthY = GroundTruthY.reshape([-1,chunck_size])
            Ground_truth_dict = {x:GroundTruthInput,y:GroundTruthY}                                   #Feed dictionary for the ground truth
            GroundTruthInput = np.repeat(GroundTruthInput,16,axis=0)                                   #16 * n_chuncks * 1
            for instance in range(Candidate_images_prob.shape[0]):
                for i in range(GroundTruthInput.shape[1]-1):
                    if GroundTruthInput[instance][i] == [0]:
                        Candidate_images_prob[instance][i] = 1-Candidate_images_prob[instance][i]                #Assign the probability of 0 to the 0 digits
            Candidate_images_prob = np.reshape(Candidate_images_prob,[16,n_chunks-1])
            Sum_log_prob = np.sum(np.log(Candidate_images_prob),axis=1)
            Best_fill_index = np.argmax(Sum_log_prob)             
            InPainting_Cross_entropy = -Sum_log_prob[Best_fill_index]       #This is essentially the cross-entropy
            digit_Best_fill = bin(Best_fill_index)
            Fill_in_digit_Collection.append(digit_Best_fill)
            Fill_in_index_collection.append(Best_fill_index)
            #Cross-entropy of the Ground Truth
            Groundtruth_prob = prediction_eval.eval(Ground_truth_dict)
            Groundtruth_prob = np.transpose(np.reshape(Groundtruth_prob,[n_chunks-1,1,1]),[1,0,2])     #1 * n_chunks-1 * 1
            Ground_truth_prediction_image = np.transpose(np.reshape(predict_result.eval(Ground_truth_dict),[n_chunks-1,1,1]),[1,0,2])  # 1*n_chunck * 1
            cross_entropy_ground_truth_list = -(np.multiply(Ground_truth_prediction_image,np.log(Groundtruth_prob)) + np.multiply(1-Ground_truth_prediction_image,np.log(1-Groundtruth_prob)))
            #print(np.shape(cross_entropy_ground_truth))
            cross_entropy_ground_truth = np.asscalar(np.reshape(np.sum(cross_entropy_ground_truth_list,axis=1),[1]))
            InPainting_Cross_entropy_Collection.append(InPainting_Cross_entropy)
            GroundTruth_Cross_entropy_Collection.append(cross_entropy_ground_truth)
            #Update to the next image
            ImageIndex = ImageIndex + 1
            print('Finished:%d',(ImageIndex*1.0)/10,'%')
        #Change Point 3 (Optional for demo, but neccessary if want to save data in the correct file)
        print('The Cross entropy of the 1000 In-painting iamges is',InPainting_Cross_entropy_Collection)
        print('The Cross entropy of the 1000 Ground-truth iamges is',GroundTruth_Cross_entropy_Collection)
        np.save('../Plot/Data/Question3_WindowMising_LSTM_Heiarchy/InPainting_Cross_entropy_Collection',np.asarray(InPainting_Cross_entropy_Collection))
        np.save('../Plot/Data/Question3_WindowMising_LSTM_Heiarchy/GroundTruth_Cross_entropy_Collection',np.asarray(GroundTruth_Cross_entropy_Collection))
        np.save('../Plot/Data/Question3_WindowMising_LSTM_Heiarchy/BestFillIndex',np.asarray(Fill_in_digit_Collection))
        np.save('../Plot/Data/Question3_WindowMising_LSTM_Heiarchy/PredictedImages',np.asarray(Predicted_Image_Collection))
        np.save('../Plot/Data/Question3_WindowMising_LSTM_Heiarchy/FillInIndex',np.asarray(Fill_in_index_collection))

#Implement above functions
WindowPinxelInPainting(x)
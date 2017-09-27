# RNN_sequence_image_prediction_and_in-painting <br />
This is a project which developed two kinds of Recurrent Neural Networks to carry out different tasks. The first kind of RNN is designed to predict the MNIST image with sequence feed-in, the second kind of RNN is designed to in-paint pixels (predict individual pixels). <br />
Usage: <br />
	1.Single layer Image Prediction with LSTM/GRU (Task1):<br />
	Run 'ImagePrediction_Implement_Single_Layer_LSTM' to test Image prediction task with single layer RNN. If you want to change the number of units, you should change the variable 'Flag_sigle_layer_LSTM'. <br />
	For the usage of GRU it is the same with the method of LSTM. <br />
	
	2.Stacked RNN  Image Prediction with LSTM/GRU (Task1):<br />
	Run 'ImagePrediction_Implement_stack_LSTM' to test the 3-layer 32-units-each LSTM. And it is the same to GRU.<br />

	3.Single layer Pixel Prediction with LSTM (Task2):<br />
	Run 'PixelPrediction_sigle_layer_Implement' to test the Pixel-wise prediction with single layer LSTM RNN. Again if you want to change the units variable 'Pixel_Prediction_Flag'
	should be changed.<br />

	4.Stacked RNN Pixel Prediction with LSTM(Task2):<br />
	Run 'PixelPrediction_stack_Implement' to test the 3-layer LSTM for pixel prediction task.<br />

	5.In-painting (Task2):<br />
	Run the 'InpaintingPrediction' for multi-step prediction and 'InpaintingPrediction_1_step' for single-step prediction. Notice that if you want to change the prediction
	steps you can simply change the related varaible. But if you want to change the restored model the RNN graph could also be changed with the correponding settings.<br />
	N.B.: The 300 masked data is provided in the dictionary 'CorruptData' and it should be there accroding to my codes.<br />
	 
	6.Missing Pixel Prediction (Task3):<br />
	Run 'MissingOnePixelInpainting' and 'MissingWindowPixelInpainting'to inspect the result. Again if the model is replaced with another then the size of RNN should
	also be changed (change variable 'rnn_size').<br />
	N.B.: I didn't attach the missing_pixel files. Please if you want to run them put then into the 'CorruptData' File.<br />

	7.Drawing Codes(Task1+2+3):<br />
	The codes I programed to draw the figures are provided in the dictionary 'Painting'.<br />

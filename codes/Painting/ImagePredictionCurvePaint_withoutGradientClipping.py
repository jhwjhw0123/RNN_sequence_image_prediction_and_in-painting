import numpy as np
import matplotlib.pyplot as plt

Lstm32_Train_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TrainAccuracy_32_tf_1_0.npy')
Lstm32_Test_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TestAccuracy_32_tf_1_0.npy')

Lstm64_Train_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TrainAccuracy_64_tf_1_0.npy')
Lstm64_Test_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TestAccuracy_64_tf_1_0.npy')

Lstm128_Train_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TrainAccuracy_128_tf_1_0.npy')
Lstm128_Test_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TestAccuracy_128_tf_1_0.npy')

Lstm_Heiarchy_Train_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TrainAccuracy_Heiarchy_tf_1_0.npy')
Lstm_Heiarchy_Test_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TestAccuracy_Heiarchy_tf_1_0.npy')

plt.figure(1)
plt.title("Trainning Accuracy Curves for 4 kinds of LSTM cells")
#32 Units lstm
plt.subplot(2,2,1)
plt.plot(Lstm32_Train_Accuracy, 'b')
plt.xlabel('epochs for 32 units LSTM')
plt.ylabel('accuracy rate')
#64 units lstm
plt.subplot(2,2,2)
plt.plot(Lstm64_Train_Accuracy, 'b')
plt.xlabel('epochs for 64 units LSTM')
plt.ylabel('accuracy rate')
#128 units lstm
plt.subplot(2,2,3)
plt.plot(Lstm64_Train_Accuracy, 'b')
plt.xlabel('epochs for 128 units LSTM')
plt.ylabel('accuracy rate')
#3 layer heiarchy lstm
plt.subplot(2,2,4)
plt.plot(Lstm_Heiarchy_Train_Accuracy, 'b')
plt.xlabel('epochs for stack LSTM')
plt.ylabel('accuracy rate')
plt.show()
plt.savefig('../Plot/Figures/LSTM_Training_Accuracy_Curve.png')

plt.figure(2)
plt.title("Test Accuracy Curves for 4 kinds of LSTM cells")
#32 Units lstm
plt.subplot(2,2,1)
plt.plot(Lstm32_Test_Accuracy, 'g')
plt.xlabel('epochs for 32 units LSTM')
plt.ylabel('accuracy rate')
#64 units lstm
plt.subplot(2,2,2)
plt.plot(Lstm64_Test_Accuracy, 'g')
plt.xlabel('epochs for 64 units LSTM')
plt.ylabel('accuracy rate')
#128 units lstm
plt.subplot(2,2,3)
plt.plot(Lstm64_Test_Accuracy, 'g')
plt.xlabel('epochs for 128 units LSTM')
plt.ylabel('accuracy rate')
#3 layer heiarchy lstm
plt.subplot(2,2,4)
plt.plot(Lstm_Heiarchy_Test_Accuracy, 'g')
plt.xlabel('epochs for stack LSTM')
plt.ylabel('accuracy rate')
plt.show()
plt.savefig('../Plot/Figures/LSTM_Test_Accuracy_Curve.png')
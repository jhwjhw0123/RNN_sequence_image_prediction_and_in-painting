import numpy as np
import matplotlib.pyplot as plt

GRU32_Train_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TrainAccuracy_32_GRU.npy')
GRU32_Test_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TestAccuracy_32_GRU.npy')

GRU64_Train_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TrainAccuracy_64_GRU.npy')
GRU64_Test_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TestAccuracy_64_GRU.npy')

GRU128_Train_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TrainAccuracy_128_GRU.npy')
GRU128_Test_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TestAccuracy_128_GRU.npy')

GRU_Heiarchy_Train_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TrainAccuracy_Heiarchy_GRU.npy')
GRU_Heiarchy_Test_Accuracy = np.load('../Plot/Data/Question1_TF1.0/TestAccuracy_Heiarchy_GRU.npy')

plt.figure(1)
plt.title("Trainning Accuracy Curves for 4 kinds of GRU cells")
#32 Units lstm
plt.subplot(2,2,1)
plt.plot(GRU32_Train_Accuracy, 'r')
plt.xlabel('epochs for 32 units GUR')
plt.ylabel('accuracy rate')
#64 units lstm
plt.subplot(2,2,2)
plt.plot(GRU64_Train_Accuracy, 'r')
plt.xlabel('epochs for 64 units GRU')
plt.ylabel('accuracy rate')
#128 units lstm
plt.subplot(2,2,3)
plt.plot(GRU128_Train_Accuracy, 'r')
plt.xlabel('epochs for 128 units GRU')
plt.ylabel('accuracy rate')
#3 layer heiarchy lstm
plt.subplot(2,2,4)
plt.plot(GRU_Heiarchy_Train_Accuracy, 'r')
plt.xlabel('epochs for stack GRU')
plt.ylabel('accuracy rate')
plt.show()
plt.savefig('../Plot/Figures/GRU_Training_Accuracy_Curve.png')

plt.figure(2)
plt.title("Test Accuracy Curves for 4 kinds of GRU cells")
#32 Units lstm
plt.subplot(2,2,1)
plt.plot(GRU32_Test_Accuracy, 'c')
plt.xlabel('epochs for 32 units LSTM')
plt.ylabel('accuracy rate')
#64 units lstm
plt.subplot(2,2,2)
plt.plot(GRU64_Test_Accuracy, 'c')
plt.xlabel('epochs for 64 units LSTM')
plt.ylabel('accuracy rate')
#128 units lstm
plt.subplot(2,2,3)
plt.plot(GRU128_Test_Accuracy, 'c')
plt.xlabel('epochs for 128 units LSTM')
plt.ylabel('accuracy rate')
#3 layer heiarchy lstm
plt.subplot(2,2,4)
plt.plot(GRU_Heiarchy_Test_Accuracy, 'c')
plt.xlabel('epochs for stack LSTM')
plt.ylabel('accuracy rate')
plt.show()
plt.savefig('../Plot/Figures/LSTM_Test_Accuracy_Curve.png')
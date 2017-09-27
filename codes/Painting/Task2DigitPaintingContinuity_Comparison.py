import numpy as np
import matplotlib.pyplot as plt

Masked_Image = np.load('../CorruptData/CorrputLast300.npy')
Ground_Truth = np.load('../CorruptData/GroundTrust.npy')
Recontrust_32 = []
##Data with LSTM 32 units
Recontrust_32.append(np.load('../Plot/Data/Question2_Inpainting_LSTM32/Reconstructed_Image_1.npy'))
Recontrust_32.append(np.load('../Plot/Data/Question2_Inpainting_LSTM32/Reconstructed_Image_10.npy'))
Recontrust_32.append(np.load('../Plot/Data/Question2_Inpainting_LSTM32/Reconstructed_Image_28.npy'))
Recontrust_32.append(np.load('../Plot/Data/Question2_Inpainting_LSTM32/Reconstructed_Image_300.npy'))
#Corresponding cross-entropy
InpaintingEntropy_32 = []
InpaintingEntropy_32.append(np.load('../Plot/Data/Question2_Inpainting_LSTM32/Holistic_Inpainting_entropy_1_step.npy'))
InpaintingEntropy_32.append(np.load('../Plot/Data/Question2_Inpainting_LSTM32/Holistic_Inpainting_entropy_10.npy'))
InpaintingEntropy_32.append(np.load('../Plot/Data/Question2_Inpainting_LSTM32/Holistic_Inpainting_entropy_28.npy'))
InpaintingEntropy_32.append(np.load('../Plot/Data/Question2_Inpainting_LSTM32/Holistic_Inpainting_entropy_300.npy'))
##Data with LSTM 64 units
Recontrust_64 = []
Recontrust_64.append(np.load('../Plot/Data/Question2_Inpainting_LSTM64/Reconstructed_Image_1.npy'))
Recontrust_64.append(np.load('../Plot/Data/Question2_Inpainting_LSTM64/Reconstructed_Image_10.npy'))
Recontrust_64.append(np.load('../Plot/Data/Question2_Inpainting_LSTM64/Reconstructed_Image_28.npy'))
Recontrust_64.append(np.load('../Plot/Data/Question2_Inpainting_LSTM64/Reconstructed_Image_300.npy'))

#Corresponding cross-entropy
InpaintingEntropy_64 = []
InpaintingEntropy_64.append(np.load('../Plot/Data/Question2_Inpainting_LSTM64/Holistic_Inpainting_entropy_1_step.npy'))
InpaintingEntropy_64.append(np.load('../Plot/Data/Question2_Inpainting_LSTM64/Holistic_Inpainting_entropy_10.npy'))
InpaintingEntropy_64.append(np.load('../Plot/Data/Question2_Inpainting_LSTM64/Holistic_Inpainting_entropy_28.npy'))
InpaintingEntropy_64.append(np.load('../Plot/Data/Question2_Inpainting_LSTM64/Holistic_Inpainting_entropy_300.npy'))
##Data with LSTM 128 units
Recontrust_128 = []
Recontrust_128.append(np.load('../Plot/Data/Question2_Inpainting_LSTM128/Reconstructed_Image_1.npy'))
Recontrust_128.append(np.load('../Plot/Data/Question2_Inpainting_LSTM128/Reconstructed_Image_10.npy'))
Recontrust_128.append(np.load('../Plot/Data/Question2_Inpainting_LSTM128/Reconstructed_Image_28.npy'))
Recontrust_128.append(np.load('../Plot/Data/Question2_Inpainting_LSTM128/Reconstructed_Image_300.npy'))
print(np.shape(Recontrust_128[2]))
#Corresponding cross-entropy
InpaintingEntropy_128 = []
InpaintingEntropy_128.append(np.load('../Plot/Data/Question2_Inpainting_LSTM128/Holistic_Inpainting_entropy_1_step.npy'))
InpaintingEntropy_128.append(np.load('../Plot/Data/Question2_Inpainting_LSTM128/Holistic_Inpainting_entropy_10.npy'))
InpaintingEntropy_128.append(np.load('../Plot/Data/Question2_Inpainting_LSTM128/Holistic_Inpainting_entropy_28.npy'))
InpaintingEntropy_128.append(np.load('../Plot/Data/Question2_Inpainting_LSTM128/Holistic_Inpainting_entropy_300.npy'))
##Data with LSTM 128 units
Recontrust_stack = []
Recontrust_stack.append(np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Reconstructed_Image_1.npy'))
Recontrust_stack.append(np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Reconstructed_Image_10.npy'))
Recontrust_stack.append(np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Reconstructed_Image_28.npy'))
Recontrust_stack.append(np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Reconstructed_Image_300.npy'))
#Corresponding cross-entropy
InpaintingEntropy_stack = []
InpaintingEntropy_stack.append(np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Holistic_Inpainting_entropy_1_step.npy'))
InpaintingEntropy_stack.append(np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Holistic_Inpainting_entropy_10.npy'))
InpaintingEntropy_stack.append(np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Holistic_Inpainting_entropy_28.npy'))
InpaintingEntropy_stack.append(np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Holistic_Inpainting_entropy_300.npy'))
#Pick them into one list
Reconstrust_image = [Recontrust_32,Recontrust_64,Recontrust_128,Recontrust_stack]
InpaintingEntropy = [InpaintingEntropy_32,InpaintingEntropy_64,InpaintingEntropy_128,InpaintingEntropy_stack]
#print(np.shape(Reconstrust_image[0][0]))
Image_Index = 91
modelname = ['32','64','128','stack_32']

fig = plt.figure(figsize=(12, 6))
for models in range(4):
    for images in range(5):
        if images == 0:
            This_image_show = np.reshape(Masked_Image[Image_Index],[28,28])
        else:
            This_image = Reconstrust_image[models][images-1]
            This_image_show = np.reshape(This_image[Image_Index],[28,28])
            This_entropy = InpaintingEntropy[models][images-1]
            print(This_entropy[Image_Index])
        ax = fig.add_subplot(4,5,models*5+images+1)
        plt.imshow(This_image_show)
        if images == 0:
            ax.set_title('Masked_Image_Image',fontsize=9)
        elif images == 1:
            ax.set_title('1 step LSTM ' + modelname[models] + ' units',fontsize=9)
        elif images == 2:
            ax.set_title('10 step LSTM ' + modelname[models] + ' units',fontsize=9)
        elif images == 3:
            ax.set_title('28 step LSTM' + modelname[models] + ' units',fontsize=9)
        elif images == 4:
            ax.set_title('300 step LSTM' + modelname[models] + ' units',fontsize=9)
plt.show()

# #Data with
# Recontrust300 = np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Reconstructed_Image_300.npy')
#
# All_sample = np.load('../Plot/Data/Question2_Inpainting_LSTM_Heiarchy/Sampled_Image_300.npy')
#
#
#
# print(Recontrust300.shape)
# print(All_sample.shape)
# GroundTruth_show = np.reshape(Ground_Truth[Image_Index],[28,28])
# Masked_Show = np.reshape(Masked_Image[Image_Index],[28,28])
# Reconstruct300_show = np.reshape((Recontrust300[Image_Index]),[28,28])
# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(131)
# ax.set_title('Original_Image')
# plt.imshow(GroundTruth_show)
# ax.set_aspect('equal')
# bx = fig.add_subplot(132)
# bx.set_title('Masked_Image')
# plt.imshow(Masked_Show)
# bx.set_aspect('equal')
# cx = fig.add_subplot(133)
# cx.set_title('Reconstructed_Image')
# plt.imshow(Reconstruct300_show)
# cx.set_aspect('equal')
# cax = fig.add_axes([0.42, 0.1, 1.08, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# # plt.colorbar(orientation='vertical')
# plt.show()
#
# fig2 = plt.figure(figsize=(6, 3.2))
# Plot_Sample_images = All_sample[Image_Index*10:(Image_Index+1)*10]
# print(np.shape(Plot_Sample_images))
# for i in range(10):
#     This_Plot = np.reshape(Plot_Sample_images[i],[28,28])
#     fig2.add_subplot(2,5,i+1)
#     ax.set_title('Sample_Image')
#     plt.imshow(This_Plot)
#     ax.set_aspect('equal')
# # plt.colorbar(orientation='vertical')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

OnepixelMissingPackage = np.load('../CorruptData/one_pixel_inpainting.npy')
ReconstructedData32 = np.load('../Plot/Data/Question3_OneMising_LSTM32/PredictedImages.npy')
ReconstructedData64 = np.load('../Plot/Data/Question3_OneMising_LSTM64/PredictedImages.npy')
ReconstructedData128 = np.load('../Plot/Data/Question3_OneMising_LSTM128/PredictedImages.npy')
ReconstructedDatastack = np.load('../Plot/Data/Question3_OneMising_LSTM_Heiarchy/PredictedImages.npy')
WindowPixel_Image_index32 = np.load('../Plot/Data/Question3_OneMising_LSTM32/BestFillIndex.npy')
WindowPixel_Image_index64 = np.load('../Plot/Data/Question3_OneMising_LSTM64/BestFillIndex.npy')
WindowPixel_Image_index128 = np.load('../Plot/Data/Question3_OneMising_LSTM128/BestFillIndex.npy')
WindowPixel_Image_indexstack = np.load('../Plot/Data/Question3_OneMising_LSTM_Heiarchy/BestFillIndex.npy')
Cross_Entropy_ground_truth = np.load('../Plot/Data/Question3_OneMising_LSTM32/GroundTruth_Cross_entropy_Collection.npy')
Cross_Entropy_In_Painting32 = np.load('../Plot/Data/Question3_OneMising_LSTM32/InPainting_Cross_entropy_Collection.npy')
Cross_Entropy_In_Painting64 = np.load('../Plot/Data/Question3_OneMising_LSTM64/InPainting_Cross_entropy_Collection.npy')
Cross_Entropy_In_Painting128 = np.load('../Plot/Data/Question3_OneMising_LSTM128/InPainting_Cross_entropy_Collection.npy')
Cross_Entropy_In_Painting_stack = np.load('../Plot/Data/Question3_OneMising_LSTM_Heiarchy/InPainting_Cross_entropy_Collection.npy')

ImageIndex = 562  #0-999

# print(np.shape(OnePixel_Image_index))
# print(np.shape(ReconstructedData))

Oringinal_Image = np.reshape(OnepixelMissingPackage[1][ImageIndex],[28,28])
Pixel_Missing_image = np.reshape(OnepixelMissingPackage[0][ImageIndex],[28,28])
#32
This_selected_image_index_32 = WindowPixel_Image_index32[ImageIndex]
Reconstruted_image32 = np.reshape(ReconstructedData32[ImageIndex][This_selected_image_index_32],[28,28])
#64
This_selected_image_index_64 = WindowPixel_Image_index64[ImageIndex]
Reconstruted_image64 = np.reshape(ReconstructedData64[ImageIndex][This_selected_image_index_64],[28,28])
#128
This_selected_image_index_128 = WindowPixel_Image_index128[ImageIndex]
Reconstruted_image128 = np.reshape(ReconstructedData128[ImageIndex][This_selected_image_index_128],[28,28])
#Stack
This_selected_image_index_stack = WindowPixel_Image_indexstack[ImageIndex]
Reconstruted_image_stack = np.reshape(ReconstructedDatastack[ImageIndex][This_selected_image_index_stack],[28,28])

# print(This_selected_image_ind)
# Binary_combination_show = bin(This_selected_image_ind)[2:]
# Binary_combination_show = Binary_combination_show.zfill(4)
fig = plt.figure(figsize=(6, 3.2))
#plt.text(0,0,'The conbination of the pixels is:'+Binary_combination_show)
ax = fig.add_subplot(231)
ax.set_title('Original_Image')
plt.imshow(Oringinal_Image)
plt.text(-18,15,'Groundturth Cross_entropy\n='+str(Cross_Entropy_ground_truth[ImageIndex]))
ax.set_aspect('equal')
bx = fig.add_subplot(232)
bx.set_title('Pixel_Missing_Image')
plt.imshow(Pixel_Missing_image)
bx.set_aspect('equal')
cx = fig.add_subplot(233)
cx.set_title('Reconstructed_Image_32')
plt.imshow(Reconstruted_image32)
plt.text(30,15,'Inpainting Cross_entropy\n='+str(Cross_Entropy_In_Painting32[ImageIndex]))
cx.set_aspect('equal')
dx = fig.add_subplot(234)
dx.set_title('Reconstructed_Image_64')
plt.imshow(Reconstruted_image64)
plt.text(5,35,'Inpainting Cross_entropy\n='+str(Cross_Entropy_In_Painting64[ImageIndex]))
dx.set_aspect('equal')
dx = fig.add_subplot(235)
dx.set_title('Reconstructed_Image_128')
plt.imshow(Reconstruted_image128)
plt.text(5,35,'Inpainting Cross_entropy\n='+str(Cross_Entropy_In_Painting128[ImageIndex]))
dx.set_aspect('equal')
dx = fig.add_subplot(236)
dx.set_title('Reconstructed_Image_stacked')
plt.imshow(Reconstruted_image_stack)
plt.text(5,35,'Inpainting Cross_entropy\n='+str(Cross_Entropy_In_Painting_stack[ImageIndex]))
dx.set_aspect('equal')

cax = fig.add_axes([0.42, 0.1, 1.08, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
# plt.colorbar(orientation='vertical')
plt.show()
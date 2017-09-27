import numpy as np
import matplotlib.pyplot as plt

OnepixelMissingPackage = np.load('../CorruptData/one_pixel_inpainting.npy')
ReconstructedData = np.load('../Plot/Data/Question3_OneMising_LSTM64/PredictedImages.npy')
OnePixel_Image_index = np.load('../Plot/Data/Question3_OneMising_LSTM64/BestFillIndex.npy')
Cross_Entropy_ground_truth = np.load('../Plot/Data/Question3_OneMising_LSTM64/GroundTruth_Cross_entropy_Collection.npy')
Cross_Entropy_In_Painting = np.load('../Plot/Data/Question3_OneMising_LSTM64/InPainting_Cross_entropy_Collection.npy')

ImageIndex = 0  #0-999

Oringinal_Image = np.reshape(OnepixelMissingPackage[1][ImageIndex],[28,28])
Pixel_Missing_image = np.reshape(OnepixelMissingPackage[0][ImageIndex],[28,28])
This_selected_image_ind = OnePixel_Image_index[ImageIndex]
Reconstruted_image = np.reshape(ReconstructedData[ImageIndex][This_selected_image_ind],[28,28])

fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(131)
ax.set_title('Original_Image')
plt.imshow(Oringinal_Image)
plt.text(5,35,'Groundturth Cross_entropy\n='+str(Cross_Entropy_ground_truth[ImageIndex]))
ax.set_aspect('equal')
bx = fig.add_subplot(132)
bx.set_title('Pixel_Missing_Image')
plt.imshow(Pixel_Missing_image)
bx.set_aspect('equal')
cx = fig.add_subplot(133)
cx.set_title('Reconstructed_Image')
plt.imshow(Reconstruted_image)
plt.text(5,35,'Inpainting Cross_entropy\n='+str(Cross_Entropy_In_Painting[ImageIndex]))
plt.text(50, .025, r'$\mu=100,\ \sigma=15$')
cx.set_aspect('equal')
cax = fig.add_axes([0.42, 0.1, 1.08, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
# plt.colorbar(orientation='vertical')
plt.show()


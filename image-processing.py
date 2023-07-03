#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Necessaary Libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np


# In[3]:


img_path = "C:\\Users\jyotp\OneDrive\Desktop\image-processing\manc.jpeg"
img = cv2.imread(img_path)
print(img.shape)


# In[4]:


gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.show()
print(gray_image.shape)


# In[11]:


# Define the coordinates of the region of interest (ROI)
y = 150  # starting x coordinate of the ROI
x = 250  # starting y coordinate of the ROI
width = 300  # width of the ROI
height = 400  # height of the ROI
# Crop the image using the ROI coordinates
cropped_img = img[y:y+height, x:x+width]
# Display the cropped image using matplotlib
plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# In[12]:


(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
'''Thresholding is performed with a threshold value of 20.
Pixels with intensity values less than 20 are set to 0 (black), and pixels with
intensity values greater than or equal to 20 are set to 255 (white). '''
plt.imshow(blackAndWhiteImage, cmap='gray')
plt.show()
(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
plt.imshow(blackAndWhiteImage, cmap='gray')
plt.show()
(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 160, 255, cv2.THRESH_BINARY)
plt.imshow(blackAndWhiteImage, cmap='gray')
plt.show()
(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
plt.imshow(blackAndWhiteImage, cmap='gray')
plt.show()


# In[13]:


output2 = cv2.blur(gray_image, (10, 10)) #Using the inbuilt Bluring function
plt.imshow(output2, cmap='gray')
plt.show()


# In[14]:


output2 = cv2.GaussianBlur(gray_image, (9, 9), 10)   #(9,9) -> represents the kernel size and 10 represents standard deviation
plt.imshow(output2, cmap='gray')
plt.show()


# In[15]:


# Define the rotation angle and scale factor
angle = 30
scale = 1.0
# Get the image dimensions
(h, w) = img.shape[:2]
# Calculate the center point of the image
center = (w // 2, h // 2)
# Define the rotation matrix
M = cv2.getRotationMatrix2D(center, angle, scale)
# Apply the rotation to the image
rotated_img = cv2.warpAffine(gray_image, M, (w, h))
# Display the rotated image
plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# In[16]:


img = cv2.imread(img_path)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, output2) = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
output2 = cv2.GaussianBlur(output2, (5, 5), 3)
output2 = cv2.Canny(output2, 180, 255)
'''It applies the Canny edge detection algorithm, which identifies edges based on gradient intensity.
The specified thresholds of 180 and 255 determine the minimum and maximum
gradient values for an edge to be considered.'''
plt.imshow(output2,cmap='gray')
plt.show()


# In[17]:


lines = cv2.HoughLinesP(output2, 1, np.pi/180,30)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),4)
plt.imshow(img)


# In[18]:


from skimage.io import imread  # pip install scikit-image
image = imread("C:\\Users\jyotp\OneDrive\Desktop\image-processing\manc.jpeg")
type(image)


# In[20]:


# Printing Image in the form of Matrix
import numpy as np
image = imread("C:\\Users\jyotp\OneDrive\Desktop\image-processing\manc.jpeg")
image_matrix = np.array(image)
print("Image matrix:")
print(image_matrix)


# In[22]:


image.shape


# In[23]:


image.ndim


# In[24]:


image.size


# In[25]:


import matplotlib.pyplot as plt
red = image[:, :, 0]
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(red, cmap='Reds_r')
plt.title("Red Channel of the Image")
plt.show()


# In[26]:


import matplotlib.pyplot as plt
green = image[:, :, 1]
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(red, cmap='Greens_r')
plt.title("Green Channel of the Image")
plt.show()


# In[27]:


import matplotlib.pyplot as plt
Blue = image[:, :, 2]
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(red, cmap='Blues_r')
plt.title("Blue Channel of the Image")
plt.show()


# In[28]:


plt.imshow(image)
plt.axis('on')  # Enable the axis
plt.show()


# In[30]:


import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread
def compare(image1, image2, title1, title2):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
image = imread("C:\\Users\jyotp\OneDrive\Desktop\image-processing\manc.jpeg")
gray = rgb2gray(image)
compare(image, gray, "Original Image", "Grayscale Image")


# In[31]:


import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
def compare(image1, image2, title1, title2):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
Rose = imread("C:\\Users\jyotp\OneDrive\Desktop\image-processing\manc.jpeg")
horizontal_flipped = np.fliplr(Rose)
vertical_flipped = np.flipud(Rose)
compare(Rose, horizontal_flipped, "Original Image", "Horizontally Flipped Image")
compare(Rose, vertical_flipped, "Original Image", "Vertically Flipped Image")


# In[32]:


def plot_with_hist_channel(image, channel):
    channels = ["red", "green", "blue"]
    channel_idx = channels.index(channel)
    color = channels[channel_idx]
    extracted_channel = image[:, :, channel_idx]
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, figsize=(9, 5)
    )
    ax1.imshow(image)
    ax1.axis("off")
    ax2.hist(extracted_channel.ravel(), bins=256, color=color)
    ax2.set_xlim([0, 100])
    ax2.set_title(f"{channels[channel_idx]} histogram")


# In[33]:


colorful_scenery = imread("C:\\Users\jyotp\OneDrive\Desktop\image-processing\manc.jpeg")
plot_with_hist_channel(colorful_scenery, "red")
plot_with_hist_channel(colorful_scenery, "green")
plot_with_hist_channel(colorful_scenery, "blue")


# In[ ]:





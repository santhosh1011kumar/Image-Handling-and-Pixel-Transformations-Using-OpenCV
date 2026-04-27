#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("Name: SANTHOSH KUMAR A")
print("Reg No: 212224230250")
import cv2
import numpy as np
import matplotlib.pyplot as plt
img =cv2.imread("C:\\Users\\admin\\Downloads\\Eagle_in_Flight.jpg",cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[3]:


img.shape


# In[4]:


img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.show()


# In[6]:


img=cv2.imread("C:\\Users\\admin\\Downloads\\Eagle_in_Flight.jpg")
cv2.imwrite("C:\\Users\\admin\\Downloads\\Eagle.png",img)


# In[7]:


img=cv2.imread("C:\\Users\\admin\\Downloads\\Eagle.png")
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[8]:


plt.imshow(img)
plt.show()
img.shape


# In[9]:


cr = img_rgb[200:400,200:400]
plt.imshow(cr)
plt.title("Cropped Region")
plt.axis("off")
plt.show()
cr.shape

print("Name: SANTHOSH KUMAR A")
print("Roll No: 212224230250")


# In[10]:


res= cv2.resize(cr,(200*2, 200*2))


# In[11]:


flip= cv2.flip(res,1)
plt.imshow(flip)
plt.title("Flipped Horizontally")
plt.axis("off")


# In[15]:


img=cv2.imread("C:\\Users\\admin\\Downloads\\Apollo-11-launch.jpg",cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb.shape


# In[16]:


text = cv2.putText(img_rgb, "Apollo 11 Saturn V Launch, July 16, 1969", (300, 700),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
plt.imshow(text, cmap='gray')
plt.title("New image")
plt.show()


# In[17]:


rcol= (255, 0, 255)
cv2.rectangle(img_rgb, (400, 100), (800, 650), rcol, 3)


# In[18]:


plt.title("Annotated image")
plt.imshow(img_rgb)
plt.show()


# In[20]:


img =cv2.imread("C:\\Users\\admin\\Downloads\\boy.jpg",cv2.IMREAD_COLOR)
img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[21]:


m = np.ones(img_rgb.shape, dtype="uint8") * 50


# In[22]:


img_brighter = cv2.add(img_rgb, m)
img_darker = cv2.subtract(img_rgb, m)


# In[23]:


plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_brighter), plt.title("Brighter Image"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_darker), plt.title("Darker Image"), plt.axis("off")
plt.show()


# In[24]:


matrix1 = np.ones(img_rgb.shape, dtype="float32") * 1.1
matrix2 = np.ones(img_rgb.shape, dtype="float32") * 1.2
img_higher1 = cv2.multiply(img.astype("float32"), matrix1).clip(0,255).astype("uint8")
img_higher2 = cv2.multiply(img.astype("float32"), matrix2).clip(0,255).astype("uint8")


# In[25]:


plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_higher1), plt.title("Higher Contrast (1.1x)"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_higher2), plt.title("Higher Contrast (1.2x)"), plt.axis("off")
plt.show()


# In[26]:


b, g, r = cv2.split(img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(b, cmap='gray'), plt.title("Blue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(g, cmap='gray'), plt.title("Green Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(r, cmap='gray'), plt.title("Red Channel"), plt.axis("off")
plt.show()


# In[27]:


merged_rgb = cv2.merge([r, g, b])
plt.figure(figsize=(5,5))
plt.imshow(merged_rgb)
plt.title("Merged RGB Image")
plt.axis("off")
plt.show()


# In[28]:


hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(h, cmap='gray'), plt.title("Hue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(s, cmap='gray'), plt.title("Saturation Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(v, cmap='gray'), plt.title("Value Channel"), plt.axis("off")
plt.show()


# In[29]:


merged_hsv = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
combined = np.concatenate((img_rgb, merged_hsv), axis=1)
plt.figure(figsize=(10, 5))
plt.imshow(combined)
plt.title("Original Image  &  Merged HSV Image")
plt.axis("off")
plt.show()


# In[30]:


print("Name: SANTHOSH KUMAR A")
print("Roll No: 212224230250")


# In[ ]:





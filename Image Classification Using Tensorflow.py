#!/usr/bin/env python
# coding: utf-8

# # 1. Setup 

# In[6]:


#get_ipython().system('pip install tensorflow opencv-python matplotlib')


# In[2]:


import tensorflow as tf
import os


# In[3]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# In[7]:


tf.config.list_physical_devices('GPU')


# 

# # 2. Remove unwanted images from the dataset

# In[41]:


import cv2
import imghdr
from matplotlib import pyplot as plt


# In[21]:


data_dir = 'data' 
data_dir_happy = 'data/happy'
data_dir_sad = 'data/sad'


# In[22]:


os.listdir(data_dir)


# In[23]:


os.listdir(data_dir_happy)


# In[24]:


os.listdir(os.path.join(data_dir,'happy'))


# In[25]:


image_exts = ['jpeg','jpg', 'bmp', 'png']


# In[26]:


image_exts


# In[36]:


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        print(image)


# In[37]:


img = cv2.imread(os.path.join('data','happy','image20.jpeg'))


# In[39]:


img


# In[40]:


img.shape


# In[42]:


plt.imshow(img)


# In[44]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# 

# ### Remove images with size less than 10 kb and remove svg images

# In[33]:


for i in os.listdir(os.path.join(data_dir,'happy')):
    if os.path.getsize(os.path.join(data_dir_happy,i)) < 10 * 1024:
        os.remove(os.path.join(data_dir_happy,i))
    filetype = i.split(".")[-1]
    if filetype == 'svg':
        os.remove(os.path.join(data_dir_happy,i))

for i in os.listdir(os.path.join(data_dir,'sad')):
    if os.path.getsize(os.path.join(data_dir_sad,i)) < 10 * 1024:
        os.remove(os.path.join(data_dir_sad,i))
    filetype = i.split(".")[-1]
    if filetype == 'svg':
        os.remove(os.path.join(data_dir_sad,i))


# In[34]:


for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)


# 

# # 3. Load Data

# In[50]:


#tf.data.Dataset??


# In[45]:


import numpy as np
from matplotlib import pyplot as plt


# In[58]:


#tf.keras.utils.image_dataset_from_directory??


# In[77]:


data = tf.keras.utils.image_dataset_from_directory('data')


# In[78]:


data


# In[79]:


data_iterator = data.as_numpy_iterator()


# In[80]:


data_iterator


# In[81]:


batch = data_iterator.next()


# In[82]:


#batch


# In[83]:


len(batch)


# In[84]:


batch[0]


# In[85]:


batch[0].shape


# In[86]:


batch[1]


# In[99]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[100]:


scaled = batch[0] / 255


# In[101]:


scaled.max()


# In[102]:


scaled.min()


# In[ ]:





# 

# # 4. Scale Data / Preprocess Data

# In[103]:


data = data.map(lambda x,y: (x/255, y))

# This is done to normalize as we get the data through the data pipeline


# In[104]:


data.as_numpy_iterator().next()


# In[105]:


data.as_numpy_iterator().next()[0].max()


# In[106]:


scaled_iterator = data.as_numpy_iterator()


# In[107]:


scaled_iterator.next()


# In[108]:


scaled_iterator.next()[0].max()


# In[109]:


batch = scaled_iterator.next()


# In[111]:


batch[0].max()


# In[110]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])


# 

# In[ ]:





# # 5. Split Data

# In[119]:


len(data) # Total 6 batches


# In[121]:


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1) + 1


# In[122]:


train_size


# In[123]:


test_size


# In[124]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# In[126]:


len(test)


# 

# # 6. Build Model

# In[115]:


train


# In[127]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[128]:


model = Sequential()


# In[ ]:


# MaxPooling2d??


# In[129]:


model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[130]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[132]:


254/2


# In[133]:


30*30*16


# In[131]:


model.summary()


# 

# # 7. Train

# In[134]:


logdir='logs'


# In[135]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Useful to save the model or to create checkpoints


# In[ ]:


hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# In[ ]:


hist


# 

# # 8. Plot Performance

# In[5]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[ ]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# 

# # 9. Evaluate

# In[ ]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[ ]:


pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[ ]:


len(test)


# In[ ]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[ ]:


print(pre.result(), re.result(), acc.result())


# 

# # 10. Test 

# In[ ]:


import cv2


# In[ ]:


img = cv2.imread('154006829.jpg')
plt.imshow(img)
plt.show()


# In[ ]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[ ]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[ ]:


yhat


# In[ ]:


if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')


# 

# # 11. Save the Model

# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model.save(os.path.join('models','imageclassifier.h5'))


# In[ ]:


new_model = load_model('imageclassifier.h5')


# In[ ]:





# In[ ]:





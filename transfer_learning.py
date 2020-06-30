import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import numpy as np

local_zip = '/content/drive/My Drive/ai/st.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/st')
local_zip = '/content/drive/My Drive/ai/st-validation.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/st-validation')
zip_ref.close()

# Directory with our training shreyash pictures
train_shreyash_dir = os.path.join('/st/shreyash')

# Directory with our training tejas pictures
train_tejas_dir = os.path.join('/st/tejas')

# Directory with our training shreyash pictures
validation_shreyash_dir = os.path.join('/st-validation/shreyash')

# Directory with our training tejas pictures
validation_tejas_dir = os.path.join('/st-validation/tejas')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/st/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        #batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary'
        )

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/st-validation/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        #batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
        
from tensorflow.keras.applications.inception_v3 import InceptionV3

model_t=InceptionV3(input_shape=(300,300,3),
                  include_top=False,
                  weights='imagenet')
                  
model_t.trainable=False

model_t.summary()

from tensorflow.keras import Model

last_layer=model_t.get_layer('mixed10')
x=tf.keras.layers.Flatten()(last_layer.output)
x=tf.keras.layers.Dense(256,activation='relu')(x)
x=tf.keras.layers.Dropout(0.3)(x)
x=tf.keras.layers.Dense(1,activation='sigmoid')(x)
model1=Model(model_t.input,x)

model1.compile(optimizer=RMSprop(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])

train_datagen1=ImageDataGenerator(rescale=1./255.)
test_datagen1=ImageDataGenerator(rescale=1./255.)
train_generator1=train_datagen1.flow_from_directory('/st/',
                                                    class_mode='binary',
                                                    target_size=(300,300))
test_generator1=test_datagen1.flow_from_directory('/st-validation/',
                                                  class_mode='binary',
                                                  target_size=(300,300))
                                                  
history=model1.fit(train_generator1,epochs=12)

test_loss,test_accuracy=model1.evaluate(test_generator1)

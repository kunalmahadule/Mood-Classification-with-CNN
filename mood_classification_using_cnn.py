# 10 May 2025
# Mood Classification using CNN Happy/Sad
'''
ImageDataGenerator - Created by keras. in genai this library is important
ImageDataGenerator = Image + Data + Generator
eda - foundation of ml
nlp - foundation of llm
dl,nn - foundation of genai
genai - foundation of agentic ai
agentic ai - mcp,a2a 
gen ai+ agentic ai -- quantam ai
electronic device+ computer vision - robotics
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os 


img = image.load_img(r"C:\Users\GauravKunal\Desktop\DS\AI\#2 DL\#2 CNN\#1 Mood classification using CNN happy,sad\TRAINING\happy\7.jpg")

plt.imshow(img)

i1 = cv2.imread(r"C:\Users\GauravKunal\Desktop\DS\AI\#2 DL\#2 CNN\#1 Mood classification using CNN happy,sad\TRAINING\happy\7.jpg")
i1 # Image to array


i1.shape # height, width, rgb
# (168, 300, 3)


train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

'''
To scale all the images i need to divide with 255

In directory all the image is not same properties are different dimensions are different so we 
can't build neural network

'''

train_dataset = train.flow_from_directory(r"C:\Users\GauravKunal\Desktop\DS\AI\#2 DL\#2 CNN\#1 Mood classification using CNN happy,sad\TRAINING",
                                          target_size=(200,200),
                                          batch_size=4,
                                          class_mode='binary')

validation_dataset = validation.flow_from_directory(r"C:\Users\GauravKunal\Desktop\DS\AI\#2 DL\#2 CNN\#1 Mood classification using CNN happy,sad\VALIDATION",
                                          target_size=(200,200),
                                          batch_size=4,
                                          class_mode='binary')

train_dataset.class_indices # {'happy': 0, 'not happy': 1}
train_dataset.classes # array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
validation_dataset.classes # array([], dtype=int32)


'''
`Sequential` groups a linear stack of layers into a `Model`

'''

# CNN Model
model = tf.keras.models.Sequential([ # convolution and Max pool layer (we created multiple because we have multiple images
                                    tf.keras.layers.Conv2D(16,(3,3),activation="relu", input_shape =(200,200,3)),
                                     tf.keras.layers.MaxPool2D(2,2),

                                     tf.keras.layers.Conv2D(32,(3,3),activation="relu",),
                                     tf.keras.layers.MaxPool2D(2,2),

                                     tf.keras.layers.Conv2D(64,(3,3),activation="relu",),
                                     tf.keras.layers.MaxPool2D(2,2),


                                    # flatten layer
                                    tf.keras.layers.Flatten(),


                                    # fully connected layer
                                    tf.keras.layers.Dense(512, activation ='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                   ])

 

 
model.compile(loss='binary_crossentropy', # for binary class classification
              optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              metrics = ['accuracy']
              )

model_fit = model.fit(train_dataset,
                      epochs =10,
                      validation_data = validation_dataset)





# Testing the model to check the images is happy or not happy

dir_path = r"C:\Users\GauravKunal\Desktop\DS\AI\#2 DL\#2 CNN\#1 Mood classification using CNN happy,sad\TESTING"

#  list the image
for i in os.listdir(dir_path):
    print(i)

# show the image
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()



dir_path = r"C:\Users\GauravKunal\Desktop\DS\AI\#2 DL\#2 CNN\#1 Mood classification using CNN happy,sad\TESTING"
for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
        
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis = 0)
    images = np.vstack([x])
    
    val = model.predict(images)
    if val == 0:
        print( ' i am happy', i.capitalize())
    else:
        print('i am not happy', i.capitalize())

 
 
 
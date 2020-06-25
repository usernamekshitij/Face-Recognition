from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

train_path = "C:/Summer Training/Face Recognition/Datasets/Train"
test_path = "C:/Summer Training/Face Recognition/Datasets/Test"

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)

for layer in vgg.layers:
    layer.trainable = False


# useful for getting number of classes
folders = glob("C:/Summer Training/Face Recognition/Datasets/Train/*")
print(len(folders))
# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)


prediction = Dense(len(folders), activation="softmax")(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    "C:/Summer Training/Face Recognition/Datasets/Train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

test_set = train_datagen.flow_from_directory(
    "C:/Summer Training/Face Recognition/Datasets/Test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)


r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
)

plt.plot(r.history["loss"], label="train loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend()
plt.show()
plt.savefig("LossVal_loss")

# accuracies
plt.plot(r.history["acc"], label="train acc")
plt.plot(r.history["val_acc"], label="val acc")
plt.legend()
plt.show()
plt.savefig("AccVal_acc")

import tensorflow as tf

from keras.models import load_model

model.save("./Face Recognition/facefeatures_new_model_5classes.h5")

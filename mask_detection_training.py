import os
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from keras.applications import MobileNetV2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense, Flatten, Input, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical

my_learning_rate = 0.0001
my_epochs = 20
my_batch_size = 32

Directory = r"E:\ML-AI\Projects\Face Mask Detection\dataset"
Categories = ["with_mask", "without_mask"]

data = []
labels = []

for c in Categories:
    path = os.path.join(Directory, c)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(c)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print(labels)

data = np.array(data,dtype = "float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size = 0.2, stratify=labels, random_state=1)
print("train_test_split performed")


aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

hmodel = baseModel.output
hmodel = AveragePooling2D(pool_size=(7,7))(hmodel)
hmodel = Flatten(name="flatten")(hmodel)
hmodel = Dense(128,activation="relu")(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2, activation="softmax")(hmodel)

model = Model(inputs = baseModel.input, outputs = hmodel)

for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(learning_rate=my_learning_rate, decay = my_learning_rate/my_epochs)
model.compile(loss="binary_crossentropy",optimizer = opt, metrics=["accuracy"])


fit = model.fit(aug.flow(trainX,trainY,batch_size=my_batch_size),
                steps_per_epoch=len(trainX)//my_batch_size,
                validation_data = (testX,testY),
                validation_steps = len(testX)//my_batch_size,
                epochs = my_epochs
)

predIdxs = model.predict(testX, batch_size = my_batch_size)
predIdxs = np.argmax(predIdxs, axis = 1)

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
model.save("mask_detection_model.model", save_format="h5")
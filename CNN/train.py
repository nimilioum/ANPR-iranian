from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from tensorflow import keras
import cv2 as cv
# import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import numpy
# from keras.backend.tensorflow_backend import set_session

def train_init():

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=config))
    # reads the pickle files and shuffles the data, encodes the labels and returns data and test X and y
    file = open('data.pickle', 'rb')
    data = numpy.load(file, allow_pickle=True)
    file = open('labels.pickle', 'rb')
    labels = numpy.load(file, allow_pickle=True)

    # using onehotencoder as data encoding
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    label_enc = LabelEncoder() # encoding strings into integers because OneHot does not support chars
    train_enc = numpy.array(label_enc.fit_transform(labels))
    train_enc = train_enc.reshape((-1, 1))
    train_l = enc.fit_transform(train_enc)

    data, train_l = shuffle(data, train_l)
    split_point = int(len(data) * 0.8)
    train_data = data[:split_point]
    train_label = train_l[:split_point]
    test_data = data[split_point:]
    test_label = train_l[split_point:]
    train_data = train_data.reshape(train_data.shape[0], 30, 30, 1) # reshaping data to be compatible with the model
    test_data = test_data.reshape(test_data.shape[0], 30, 30, 1)

    return train_data, train_label, test_data, test_label


def train(train_data, train_label, test_data, test_label):
    # makes a CNN model and trains data
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(30, 30, 1))) # first Conv layer
    model.add(Conv2D(64, (3, 3), activation='relu'))                                          # recurrenting Conv layer
    model.add(MaxPool2D((3, 3)))                                                              # first pool layer
    model.add(Dropout(0.25))                                                                  # dropping out to avoid overfitting

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))        # second Conv layer
    # model.add(Conv2D(64,(3,3),activation='relu'))                         # recurrenting gives an error in third layer
    model.add(MaxPool2D((3, 3)))                                            # second pool layer
    model.add(Dropout(0.25))                                                # dropping out to avoid overfitting

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))        # third Conv layer
    # model.add(Conv2D(64,(3,3),activation='relu'))                         # gives an error
    model.add(MaxPool2D((3, 3)))                                            # third pool layer
    model.add(Dropout(0.25))                                                # dropping out to avoid overfitting

    model.add(Flatten())                          # make the output 1D
    model.add(Dense(512, activation='relu'))      # a fully connected layer
    model.add(Dropout(0.5))                       # dropping out to avoid overfitting
    model.add(Dense(41, activation='softmax'))    # output layer

    # print(model.summary())
    # batch = int(train_data.shape[0] / 30)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_data, train_label, epochs=30,
                        batch_size=41, use_multiprocessing=True)
    eval = model.evaluate(test_data,test_label)
    model.save('cnn_classifier2.h5')
    return history


def result_visual(history):
    # plotting accuracy and loss in each epoch
    print(history)
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

    # with open('history.json', 'wb') as f: # writes history in a json file to use later ( not sure when :D )
    #     json.dump(bytes(history.history), f)


# triggers the training
train_data, train_label, test_data, test_label = train_init()
# i = 5433
# cv.imshow('',train_data[i])
# cv.waitKey(0)
# print(train_label[i])
history = train(train_data, train_label, test_data, test_label)
result_visual(history)

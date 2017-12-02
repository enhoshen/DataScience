import sys
import numpy as np
import pickle
from time import time

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten


def loaddata ():
    data_dir = './cifar-10-batches-py/'
    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_2', 'data_batch_2', 'data_batch_2']
    test_file = 'test_batch'
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
        # unpickle
    data = []
    for file in data_files:
            data.append(unpickle(data_dir+file))
    test = unpickle(data_dir+test_file)

    # preprocess
    data_x = []
    data_y = np.zeros((len(data),len(data[0][b'labels']),10))
    for i in range(len(data)):
            data_x.append(data[i][b'data'])
            for j in range(len(data[i][b'labels'])):
                    data_y[i][j][data[i][b'labels'][j]] = 1
    data_x = np.reshape(np.array(data_x),(-1,3,32,32))
    data_x = np.transpose(data_x, (0, 2, 3, 1))
    data_y = np.reshape(data_y,(-1,10))
    trn_x, val_x, trn_y, val_y = train_test_split(data_x, data_y, test_size=0.02)

    tst_x = np.reshape(test[b'data'],(-1,3,32,32))
    tst_x = np.transpose(tst_x, (0, 2, 3, 1))
    tst_y = np.zeros((len(test[b'labels']),10))
    for i in range(len(test[b'labels'])):
            tst_y[i][test[b'labels'][i]] = 1
    return trn_x,trn_y,val_x,val_y,tst_x,tst_y 
def conv ( model , flt_shape ):
    model.add(Convolution2D(*flt_shape , border_mode='same'  ,dim_ordering='tf',activation='relu'))
def convBN ( model, flt_shape):
    model.add(Convolution2D(*flt_shape , border_mode='same'  ,dim_ordering='tf'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
def pool ( model ):
    model.add(MaxPooling2D((2,2),dim_ordering='tf'))
def fc   ( model , flt_shape):
    model.add(Dense(flt_shape))
    model.add(Activation('relu'))
def mymodel () :
    model = Sequential()
    model.add(Convolution2D(32,3,3, border_mode='same', input_shape=(32,32,3), dim_ordering="tf"))
    model.add(Activation('relu'))
    conv ( model,(32,3,3))
    pool ( model)

    conv ( model, (64,3,3))
    conv ( model, (64,3,3))
    pool ( model)
   
    conv ( model, (128,3,3))
    conv ( model, (128,3,3))
    pool ( model)

    model.add(Flatten())
    fc   ( model, 256 )
    model.add(Dropout(0.2))
    fc   ( model, 128 )
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.0002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
def myBNmodel():
    model = Sequential()
    model.add(Convolution2D(32,3,3, border_mode='same', input_shape=(32,32,3), dim_ordering="tf"))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))
    convBN ( model,(32,3,3))
    pool ( model)


    convBN ( model, (64,3,3))
    convBN ( model, (64,3,3))
    pool ( model)
   
    convBN ( model, (128,3,3))
    convBN ( model, (128,3,3))
    pool ( model)

    model.add(Flatten())
    fc   ( model, 256 )
    model.add(Dropout(0.2))
    fc   ( model, 128 )
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.0002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
def alexmodel():
    model = Sequential()
    model.add(Convolution2D(96,3,3 , border_mode='same',input_shape=(32,32,3),dim_ordering='tf'))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))
    convBN ( model, (256,3,3))
    pool ( model)
    convBN ( model, (384,3,3))
    convBN ( model, (384,3,3))

    convBN ( model, (256,3,3))
   
    model.add(Flatten())
    fc ( model, 512)
    model.add(Dropout(0.5))
    fc ( model, 256)
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = Adam(lr=0.0002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model 
def main():
    train_x , train_y , val_x, val_y, test_x, test_y = loaddata()
    #model = mymodel()
    model = myBNmodel()
    #model = alexmodel()

    # augmentation
    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            rotation_range=10,
            zca_whitening=False,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False)
    datagen.fit(train_x)

    # train
    ite = 100
    epoch = 1
    best_loss = 1
    batch_size = 100
    
    for i in range(ite):
        print('iter: ' + str(i+1) + '/' + str(ite))
        model.fit_generator(
                datagen.flow(train_x, train_y, batch_size=batch_size),
                samples_per_epoch=train_x.shape[0],
                epochs=epoch,
                verbose=1,
                pickle_safe=False
                )
        print('validation:')
        score = model.evaluate(val_x, val_y, batch_size=batch_size)
        print('val loss: ' + str('%.3f'%score[0]) + ', val acc: ' + str('%.3f'%score[1]))


    score = model.evaluate(test_x, test_y, batch_size=batch_size)
    print('testing loss: ' + str('%.3f'%score[0]) + ', testing acc: ' + str('%.3f'%score[1]))
    # test
    print('testing loss: ' + str('%.3f'%best_score[0]) + ', testing acc: ' + str('%.3f'%best_score[1]))
        


if __name__ == '__main__':
    main()

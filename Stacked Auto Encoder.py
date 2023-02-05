from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop,Adam

batch_size = 128
epochs = 100
input_img = Input(shape = (128, 128, 1))

# build the encoder and decoder
def encoder(input_img):
  en1 = Conv2D(32, (3,3), activation ='relu', padding ='same')(input_img)
  en1 = BatchNormalization()(en1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(en1)
  drop1 = Dropout(0.4)(pool1)

  en2 = Conv2D(64, (3,3), activation='relu', padding='same')(drop1)
  en2 = BatchNormalization()(en2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(en2)
  drop2 = Dropout(0.4)(pool2)

  en3 = Conv2D(128, (3,3), activation='relu', padding='same')(drop2)
  en3 = BatchNormalization()(en3)
  
  return en3

def decoder(en3):

  de4 = Conv2D(64, (3,3), activation='relu', padding='same')(en3)
  de4 = BatchNormalization()(de4)
  unpool1 = UpSampling2D((2, 2))(de4)

  de5 = Conv2D(32, (3,3), activation='relu', padding='same')(unpool1)
  de5 = BatchNormalization()(de5)
  unpool2 = UpSampling2D((2, 2))(de5)

  decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(unpool2) 

  return decoded

autoencoder = keras.Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(optimizer= RMSprop(),loss='mean_squared_error')

autoencoder_train = autoencoder.fit(x_train,x_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,x_test))

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(2, activation='softmax')(den)
    return out

encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam',metrics=['accuracy'])

# augment the training data
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

classify_train = full_model.fit(datagen.flow(x_train,y_train, batch_size=64),epochs=100,verbose=1,validation_data=(x_test,y_test))

# test the model performance on testing set
test_eval = full_model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

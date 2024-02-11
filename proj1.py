from sklearn.decomposition import PCA
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
from keras.layers import Conv2D, Input, Flatten, Dense, Reshape, Conv2DTranspose
import keras.backend as K
from keras import Model

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
X_train1 = X_train
X_test1 = X_test


latent_dim = 16
#shape od X_train: (60000, 28, 28)
#shape od X_test: (10000, 28, 28)
#slike su matrice 28*28, a da bi mogao da se primeni PCA, moraju da se pretvore u vektore dimenzije 28*28=784 i da se normalizuju

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[1]) / 255.0
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[1]) / 255.0
#print(X_train.shape) 
#print(X_test.shape)
#shape od X_train: (60000, 784)
#shape od X_test: (10000, 784)


pca = PCA(latent_dim)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#print(X_train_pca.shape)
#print(X_test_pca.shape)
#shape od X_train_pca : (60000, 16)
#shape od X_test_pca : (10000, 16)
#ovim je smanjena velicina vektora sa 784 na 16 znacajnih komponenti

#X_original su slike dobijene inverznim pca
X_original  = pca.inverse_transform(X_test_pca)


########## DECODER #####################################################################

latent_inputs = Input(shape=(latent_dim,))
#laten_inputs - ulazi u decoder
x = Dense(units=7 * 7 * 32) (latent_inputs)
x = Reshape((7, 7, 32))(x)
x= Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
x= Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
outputs = Conv2DTranspose (filters=1, kernel_size=3, padding='same', activation='sigmoid')(x)
#ulaz u decoder su latent_inputs, izlaz je outputs - slika
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


######### TRENIRANJE DECODER-A ###################################################
decoder.compile(optimizer='adam', loss='mse')
history = decoder.fit(X_train_pca, X_train1, batch_size=128, epochs=10, verbose=1, validation_data=(X_test_pca, X_test1))

####### ISCRTAVANJE ###############################################################

X_reconstructed = decoder.predict(X_test_pca)


plt.figure(figsize=(10,10)) 
for i in range(5):
    #Prva kolona je X_test
    plt.subplot(5,4,i*4 + 1)
    plt.imshow(X_test1[i])
    #Druga kolona je slika koja je izlaz iz encoder-a
    plt.subplot(5,4,i*4+2)
    plt.imshow(X_test_pca[i].reshape(4,4))
    #Treca kolona je izlaz iz decoder-a
    plt.subplot(5,4,i*4+3)
    plt.imshow(X_reconstructed[i].reshape(28,28))
    #Cetvrta kolona je X_original sto je slika dobijena inverznim pca
    plt.subplot(5,4,i*4+4)
    plt.imshow(X_original[i].reshape(28,28))
    plt.axis('off')
    plt.show()
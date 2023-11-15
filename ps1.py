# #import required libraries
# import keras
# import laberlbinarizer 
# import numpy
# import keras.backend as K
# import matplotlib
# bla bla import mnist


from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report


#set a argument as output where user specifies output directory
#create object
ap=argparse.ArgumentParser()
#add variables andmethods for add_argument
ap.add_argument("-o","--output",required=True,help="helpheehee")
#add variables and method to parse object output and convert to dictionary
args=vars(ap.parse_args())




# #load the dataset
print("loading dataset wait a bit")

# load and put into test and train

# ((trainX,trainY),(testX,testY))=mnist.load_data()
# #traiiny and testy are single values of the digit from 0-9
# trainX=trainX.reshape(trainX.shape[0],28*28*1)
# testX=testX.reshape(testX.shape[0],28*28*1)


#IF DATASET IS GIVEN:










# #reshape the dataset and normalize between 0 and 1

# trainX=trainX.astype("float32")/255.0
# testX=testX.astype("float32")/255.0
# #originally we have grayscale image of matrix 28*28.we reshape it to 1D array of size 784
# #secondly intensity range of every pixel is from 0 to 255 so we divide it by 255 to normalize


# #use label binarizer for one hot encoding
# lb=LabelBinarizer()
# trainY=lb.fit_transform(trainY)
# testY=lb.fit_transform(testY)



import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
train_df = pd.read_csv('C://Users//Chinmayee//Desktop//DL//LP-IV-datasets//LP-IV-datasets//MNIST//mnist_train.csv')  # Update 'your_dataset.csv' with your actual file name
test_df = pd.read_csv('C://Users//Chinmayee//Desktop//DL//LP-IV-datasets//LP-IV-datasets//MNIST//mnist_test.csv')
# Step 2: Extract pixel values and labels
trainX = train_df.drop('label', axis=1).values
trainy =train_df['label'].values


testX = test_df.drop('label', axis=1).values
testy =test_df['label'].values
print(testy)


trainX=trainX.reshape((-1,28*28)).astype('float32')/255.0
testX=testX.reshape((-1,28*28)).astype('float32')/255.0



trainy = to_categorical(trainy, num_classes=10)
testy = to_categorical(testy, num_classes=10)  # Assuming labels are in the range 1-10












#start the layers model is sequential first two layers are dense 256,128,and
#10 weights respectively epochs and output will be 10 size vector

model=Sequential()
model.add(Dense(512,input_shape=(784,),activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(10,activation="softmax"))




#initialize sgd optimizer
sgd=SGD(0.01)

#model.compile and h=model .compile,loss function,ctegory entropy,sgd optimizer
print("training network....")
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])
H=model.fit(trainX,trainy,validation_data=(testX,testy),epochs=10,batch_size=128)



#evaluate model by generating a classification report

print("Evaluating network.....")
predictions=model.predict(trainX,batch_size=128)
print(classification_report(trainy.argmax(axis=1),predictions.argmax(axis=1)))


#plot the loss function and various plot and savefig

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,10),H.history["val_loss"],label="valloss")
plt.plot(np.arange(0,10),H.history["loss"],label="trainloss")
plt.plot(np.arange(0,10),H.history["accuracy"],label="trainaccuracy")
plt.plot(np.arange(0,10),H.history["val_accuracy"],label="valaccuracy")
plt.xlabel("epochs")
plt.ylabel("loss/accuracy")
plt.title("figure")
plt.legend()
plt.savefig(args["output"])


#done!
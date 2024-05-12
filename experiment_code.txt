#importing necessary libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing testing and training datasets
training_dataset = pd.read_csv('sign_mnist_train.csv')
testing_dataset = pd.read_csv('sign_mnist_test.csv')

#training labels
labels=training_dataset['label'].values

#plotting quantities in each class
plt.figure( figsize = (18,8) )
sns.countplot(x = labels)

#seperate labels from training dataset by dropping it
training_dataset.drop('label', axis = 1 , inplace = True)

#extraction of image data from every single row of csv file
imgs = training_dataset.values
print('check1',imgs.ndim)
imgs = np.array([np.reshape(i,(28,28)) for i in imgs])
print('check2',imgs.ndim)
imgs = np.array([i.flatten() for i in imgs])
print(imgs.ndim)

#hot one encoding for labels
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
labels=lb.fit_transform(labels)
#print(labels)


#inspect img
print(labels[6])
plt.imshow(imgs[6].reshape(28,28))

#reorganizing testing and training datasets
#x_train = imgs
#x_test = labels
#y_test = lb.fit_transform(testing_dataset['label'].values)
#testing_dataset.drop('label',axis =1,inplace=True)
#y_train = testing_dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(imgs,labels, test_size=0.3,random_state=101)

#importing models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

batch_size =128
num_classes = 24
epochs = 10

#scale imgs
x_train = x_train / 255
x_test = x_test / 255

#reshaping images into shapes required by tenserflow and keras
x_train  = x_train.reshape(x_train.shape[0],28,28,1)
x_test  = x_test.reshape(x_test.shape[0],28,28,1)
plt.imshow(x_train[9].reshape(28,28))

#creating neural network
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3),activation='relu',input_shape=(28,28,1) ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(num_classes,activation='softmax'))

#compiling model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

print(model.summary())


#training the model--
#ERROR OCCURS HERE PROBABLY DUE TO OLDER VERSION OF NUMPY
#UPDATING NUMPY DOESNOT RESOLVE THE ISSUE
#REDUCING DROPOUT RESOLVED THE ISSUE
history= model.fit(
    x_train,
    y_train,
    validation_data=(x_test,y_test),
    epochs=epochs,
    batch_size=batch_size
   )              
print('cheker')
#saving model
model.save("asl-project")
print('model saved')

#reshape test data to evaluate its performance on unseen data
test_labels = testing_dataset['label']
testing_dataset.drop('label', axis=1, inplace=True)

testing_images= testing_dataset.values
testing_images = np.array([np.reshape(i,(28,28)) for i in testing_images])
testing_images = np.array([i.flatten() for i in testing_images])
 
test_labels= lb.fit_transform(test_labels)

testing_images = testing_images.reshape(testing_images.shape[0],28,28,1)

testing_images.shape

y_predict = model.predict(testing_images)
#accuracy 
acc_score= (test_labels, y_predict.round())


#function to match label to letter
def getLetter(result):
    classLabels = { 0: 'A',
                   1: 'B',
                   2: 'C',
                   3: 'D',
                   4: 'E',
                   5: 'F',
                   6: 'G',
                   7: 'H',
                   8: 'I',
                   9: 'K',
                   10: 'L',
                   11: 'M',
                   12: 'N',
                   13: 'O',
                   14: 'P',
                   15: 'Q',
                   16: 'R',
                   17: 'S',
                   18: 'T',
                   19: 'U',
                   20: 'V',
                   21: 'W',
                   22: 'X',
                   23:'Y'}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"


#webcam code
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    frame=cv2.flip(frame,1)
    
    roi= frame[100:400, 320:620]
    cv2.imshow('roi',roi)
    roi=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28,28), interpolation = cv2.INTER_AREA)
    
    cv2.imshow('roi scaled and gray',roi)
    copy= frame.copy()
    cv2.rectangle(copy,(320,180),(620,400), (255,0,0), 5)
    
    roi = roi.reshape(1,28,28,1)
    
    result= str(model.predict_classes(roi,1,verbose=0)[0])
    cv2.putText(copy,getLetter(result), (300,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    cv2.imshow('frame', copy)
    
    if cv2.waitKey(1) == 13: #enter key
        break
            
cap.release()
cv2.destroyAllWindows()




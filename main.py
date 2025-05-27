import keras.optimizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, TensorBoard
import datetime

data=pd.read_csv('Churn_Modelling.csv')
data=data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
#print(data)

#                         zamiana płci na 1 i 2
label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
#print(data)

#                       onehot encode 'Geography'
onehot_encoder_geo=OneHotEncoder()
geo_encoder=onehot_encoder_geo.fit_transform(data[['Geography']])
geo_encoder.toarray()
#print(onehot_encoder_geo.get_feature_names_out(['Geography']))
geo_encoder_df=pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
#print(geo_encoder_df)

#                   połączenie wszystkich kolumn
data=pd.concat([data.drop('Geography', axis=1), geo_encoder_df], axis=1)
#print(data.head())

#                  zachowanie enkodera i scalera
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)

#                    devide dataset into independent and dependent features
X=data.drop('Exited', axis=1)
y=data['Exited']

#                    split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#                  sacle features down
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

#                      model ANN
model=Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), #first hidden layer and input
    Dense(32, activation='relu'),                                  #second hidden layer
    Dense(1, activation='sigmoid')                                 #output
])

model.summary()

#                      compile the model
opt=keras.optimizers.Adam(learning_rate=0.01)
loss=keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])

#                     set up the tensorboard
log_dir='logs/fit/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorflow_callback=TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch')

#                     set up early stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

#                     model training
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=32,callbacks=[tensorflow_callback, early_stopping_callback])
model.save('model.h5')

#                     load tensorboard extension - terminal -> tensorboard --logdir=logs/fit
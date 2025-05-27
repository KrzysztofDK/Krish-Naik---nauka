from keras.models import load_model
import pickle
import pandas as pd

#                 load ANN train model
model=load_model('model.h5')

#                 load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)

#                    example data
input_data={
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
input_df=pd.DataFrame([input_data])

#                    encode categorical variables
input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])

#                    one-hot encode 'geography'
geo_encoded=onehot_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#                    concatenate one hot encoded
input_df=pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

#                    scaling input data
input_scaled=scaler.transform(input_df)

#                    prediction
prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]
print(prediction_proba)

if prediction_proba>0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')
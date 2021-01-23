# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:07:01 2021

@author: Pedro Rosas Vernet. Contact: prosas.ds@gmail.com 
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('car_price_random_forest.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Year_log = standard_to.fit_transform(np.log(Year).reshape(-1,1))
        
        Kms_Driven = int(request.form['Kms_Driven'])
        Kms_Driven_log = standard_to.fit_transform(np.log(Kms_Driven).reshape(-1,1))
        
        Owner = int(request.form['Owner'])
        
        Fuel_Type_Petrol = request.form['Fuel_Type_Petrol']
        
        if(Fuel_Type_Petrol=='Petrol'):
                Fuel_Type_Petrol = 1
                Fuel_Type_Diesel = 0
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1
        
        Seller_Type_Individual=request.form['Seller_Type_Individual']
        if(Seller_Type_Individual == 'Individual'):
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0	
            
        Transmission_Mannual=request.form['Transmission_Mannual']
        if(Transmission_Mannual == 'Mannual'):
            Transmission_Mannual = 1
        else:
            Transmission_Mannual = 0
            
        prediction = model.predict([[Year_log, Kms_Driven_log, Owner, Fuel_Type_Diesel, Seller_Type_Individual, Transmission_Mannual]])
        output = round(np.exp(prediction[0]), 3) #reverting transformations applied 
        if output<0:
            return render_template('index.html',prediction_texts="Sorry, this car's market price cannot be estimated")
        else:
            return render_template('index.html',prediction_text="Market price for this car is around {} Indian Rupees".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)


from flask import Flask, jsonify, render_template, request
import pandas as pd
import pickle
import urllib.parse

data = pd.DataFrame(data=pd.read_csv("../../Data/Dataset_zero_numeric_missing.csv",sep=";"))
provinces = data['province'].unique()
types_of_property = data['type_of_property'].unique()
# load the model from disk
loaded_model = pickle.load(open('./model/GradientBoosting_on_dataset_zero_missing.sav', 'rb'))

app = Flask(__name__)

# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/')
def index():
    return render_template('index.html', provinces = provinces, types_of_property=types_of_property)

@app.route('/_get_cities')
def get_cities():
    a = request.args.get('province')
    return jsonify(data[data['province']==a]['city'].unique().tolist()) 

@app.route('/_predict')
def predict():
    data = request.args.get('data')
    data = data.split("&")
    X = {}
    for value in data:
        temp = value.split("=")
        X[temp[0]] = temp[1]
    X_to_pred = [X['bathrooms'], X['bedrooms'], X['education'], X['transport_and_public_services'], X['province'], X['city'], urllib.parse.unquote(X['type_of_property'])]
    y_pred = loaded_model.predict(X_to_pred)
    return X_to_pred

if __name__ == "__main__":
        app.run()

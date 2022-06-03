import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model_reg=pickle.load(open('reg_model_pkl','rb'))
model_clf=pickle.load(open('clf_model.pkl','rb'))
scalar_clf=pickle.load(open('scaler_clf.pkl','rb'))
scalar_reg=pickle.load(open('scaler_reg.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict_api_clf',methods=['POST'])
def predict_api_clf():

    data=request.json['data']
    #print(list(data.values()))
    nw=scalar_clf.transform([list(data.values())])
    print(nw)
    
    output=model_clf.predict(nw)[0]
    if output==0:
        output='Not Fire'
    else:
        output='Fire'
    return jsonify(output)

@app.route('/predict_api_reg',methods=['POST'])
def predict_api_reg():

    data=request.json['data']
    #print(list(data.values()))
    nw=scalar_reg.transform([list(data.values())])
    print(nw)
    
    output=model_reg.predict(nw)[0]
    return jsonify(float(output))


@app.route('/predict_clf',methods=['POST'])
def predict_clf():

    data=[float(x) for x in request.form.values()]
    final_features =scalar_clf.transform([data])
    print(data)
    
    output=model_clf.predict(final_features)[0]
    print(output)
    if output==0:
        output='Not Fire'
    else:
        output='Fire'
    return render_template('home.html', prediction_text1="Classes is  {}".format(output))

@app.route('/predict_reg',methods=['POST'])
def predict_reg():

    data=[float(x) for x in request.form.values()]
    final_features =scalar_reg.transform([data])
    print(data)
    
    output=model_reg.predict(final_features)[0]
    print(output)

    return render_template('home.html', prediction_text2="Fire Weather Index is  {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
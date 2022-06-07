import pickle
import re
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

app=Flask(__name__)
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app.config['UPLOAD_FOLDER']=os.getcwd()
model_reg=joblib.load(open('reg_model_pkl','rb'))
model_clf=joblib.load(open('clf_model_pkl','rb'))
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

@app.route('/predict_api_clf_bulk',methods=['POST'])
def predict_api_clf_bulk():

    data=request.json['data']
    #print(list(data.values()))
    df=pd.read_csv(data)
    nw=scalar_clf.transform(df)
    
    output=model_clf.predict(nw)
    df['Classes']=output
    df.to_csv("predict_clf.csv",index=False)
    st="check file predict_clf.csv in path "+os.getcwd()
    return st

@app.route('/predict_api_reg_bulk',methods=['POST'])
def predict_api_reg_bulk():

    data=request.json['data']
    #print(list(data.values()))
    df=pd.read_csv(data)
    nw=scalar_clf.transform(df)
    
    output=model_clf.predict(nw)
    df['FWI']=output
    df.to_csv("predict_reg.csv")
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


@app.route('/predict_clf_bulk',methods = ['POST'])
def predict_clf_bulk():
    if request.method == 'POST':
        file = request.files['file']
        print(file.filename)
        print(allowed_file(file.filename))
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(file)

        filepath=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        df=pd.read_csv(filepath)
        print(df.head())
        final_features =scalar_clf.transform(df)
        
        output=model_clf.predict(final_features)
        df['Classes']=output
        
        
        return render_template('bulk_pred.html', tables=[df.to_html()], titles=[''])
    
@app.route('/predict_reg_bulk',methods=['POST'])
def predict_reg_bulk():
    try:
        file = request.files['file']
        print(file.filename)
        print(allowed_file(file.filename))
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(file)

        filepath=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        print(filepath)
        df=pd.read_csv(filepath)
        print(df.head())
        final_features =scalar_reg.transform(df)
        
        output=model_reg.predict(final_features)
        df['FWI']=np.round(output,2)
        print(output)
       
        return render_template('bulk_pred.html', tables=[df.to_html()], titles=[''])
    except Exception as e:
        print(e)
if __name__=="__main__":
    app.run(debug=True)
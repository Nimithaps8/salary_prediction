from flask import Flask,make_response,request,render_template
import io
from io import StringIO
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)


def feature_engineering(df):
    df.columns=['age','workclas','fnlwgt','education','education-num','matrial-status','occupation','relationship','race','sex','capital-gain','captial-loss','hours-per-week','native-country']

    df=df.drop('fnlwgt',axis=1)
    df['sex']=np.where(df['sex']=="Male",0,1)
    label_enco_race={value:key for key,value in enumerate (df['race'].unique())}
    df['race']=df['race'].map(label_enco_race)

    label_enco_relationship={value:key for key,value in enumerate (df['relationship'].unique())}
    df['relationship']=df['relationship'].map(label_enco_relationship)

    df['occupation']=np.where(df['occupation']==' ? ','Missing',df['occupation'])
    label_enco_occu={value:key for key,value in enumerate(df['occupation'].unique())}
    df['occupation']=df['occupation'].map(label_enco_occu)

    label_enco_martial_status={value:key for key,value in enumerate (df['marital-status'].unique())}
    df['marital-status']=df['marital-status'].map(label_enco_martial_status)

    label_enco_education={value:key for key,value in enumerate (df['education'].unique())}
    df['education']=df['education'].map(label_enco_education)


    label_enco_workclass={value:key for key,value in enumerate (df['workclass'].unique())}
    df['workclass']=df['workclass'].map(label_enco_workclass)

    label_enco_native_country={value:key for key,value in enumerate (df['native-country'].unique())}
    df['native-country']=df['native-country'].map( label_enco_native_country)
    return df

def scalar(df):
    sc=StandardScaler()
    x=df[['age','workclass','education','education-num','martial-status','occupation','relationship','race','sex','capital-gain','captial-loss','hours-per-week','native-country']]
    x=sc.fit_transform(x)
    return (x)


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict",methods=['POST'])
def predict():
    f=request.files['data_file']
    if not f:
        return render_template('index.html',prediction_text='No file selected')
    stream=io.StringIO(f.stream.read().decode('UTF8'),newline=None)
    result=stream.read()
    df=pf.read_csv(StringIO(result))

    #feature Engineering
    df=feature_engineering(df)

    x=scaler(df)


    load_model=pickle.load(open('lg_model.pkl','rb'))


    print(loaded_model)


    result=load_model.predict(x)
    return render_template('index.html',prediction_text="Prediction Salary is/are: {}" .format(result))

if __name__=="__main__":
    app.run(debug=False,port=9000)
    


    

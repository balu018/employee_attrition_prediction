from flask import *
from pandas import *
from numpy import *
from sklearn import *
from pickle import *
import joblib
app=Flask(__name__)
model=joblib.load('model.pkl')
@app.route('/')
def hello():
    return render_template('a2.html')
@app.route('/',methods=['POST','GET'])
def predict():
    fea = [float(x) for x in request.form.values()]
    fea=[array(fea)]
    pred1=model.predict(fea)
    if pred1 == 0:
        return render_template('a2.html', name='employee has low risk to leave job')
    else:
        return render_template('a2.html', name='employee has high risk to leave job')
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)

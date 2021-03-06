# importing the  libraries
import pickle 
from flask import Flask, render_template, request

# Global Variables
loadedModel=pickle.load(open('KNN Model.pkl','rb'))
app=Flask(__name__)


#routers
@app.route('/')
def home():
    return render_template('iris.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    SepalLengthCm = request.form['SepalLengthCm']
    SepalWidthCm = request.form['SepalWidthCm']
    PetalLengthCm = request.form['PetalLengthCm']


    prediction = loadedModel.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm]])

    if prediction[0] == 0:
        prediction = "Iris-setosa"
    elif prediction == 1:
        prediction = "Iris-versicolor"
    else:
        prediction = "Iris-virginica"

    return render_template('iris.html', api_output=prediction)



#main Function
if __name__ == '__main__':
    app.run(debug=True)


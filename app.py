from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('fish.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        species = float(request.form['species'])
        length1= float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        width = float(request.form['width'])
        height = float(request.form['height'])
        features = np.array([[species,length1,length2,length3,height,width]])
        weight = model.predict(features)[0]
        result = f"Predicted Weight: {weight:.2f}"
        print("this is the result",result)
        
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)



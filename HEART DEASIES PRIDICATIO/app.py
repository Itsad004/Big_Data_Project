from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # No need to specify /template/

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])

        # Create DataFrame for prediction
        user_data = pd.DataFrame([[age, cp, thalach]],
                                 columns=['age', 'cp', 'thalach'])

        prediction = model.predict(user_data)

        result = "Heart Disease Present (beta tum to gya)" if prediction[0] == 1 else "No Heart Disease"
        return render_template('index.html', prediction_text=result, age=age, cp=cp, thalach=thalach)

    except Exception as e:
        return render_template('index.html', error_text=f"Error: {str(e)}")

if __name__ == '_main_':
    app.run(debug=True)
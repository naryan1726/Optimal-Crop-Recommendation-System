from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the voting classifier from the pickle file
with open('voting_classifier.pkl', 'rb') as file:
    voting_clf = pickle.load(file)

# Crop mapping (same as used earlier, but reversed for decoding prediction)
crop_mapping = {
    1: 'Rice', 2: 'Maize', 3: 'ChickPea', 4: 'KidneyBeans', 5: 'PigeonPeas',
    6: 'MothBeans', 7: 'MungBean', 8: 'Blackgram', 9: 'Lentil', 10: 'Pomegranate',
    11: 'Banana', 12: 'Mango', 13: 'Grapes', 14: 'Watermelon', 15: 'Muskmelon',
    16: 'Apple', 17: 'Orange', 18: 'Papaya', 19: 'Coconut', 20: 'Cotton',
    21: 'Jute', 22: 'Coffee'
}

# Initialize the Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph_value = float(request.form['ph_value'])
        rainfall = float(request.form['rainfall'])

        # Create an input array for prediction
        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])

        # Make prediction using the voting classifier (numerical value)
        crop_prediction_num = voting_clf.predict(input_features)[0]

        # Reverse map the numerical prediction to crop name
        crop_prediction = crop_mapping.get(crop_prediction_num, "Unknown Crop")

        # Render the result in the template
        return render_template('result.html', prediction=crop_prediction)
    except Exception as e:
        return f"An error occurred: {e}"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

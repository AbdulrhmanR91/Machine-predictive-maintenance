from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# Load the trained model, scaler, and label encoder
model = joblib.load('best_random_forest_model_fast.pkl')
scaler = joblib.load('scaler_fast.pkl')
label_encoder = joblib.load('label_encoder_fast.pkl')

# Define the feature columns based on training data
feature_columns = ['Type', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                    'Air temperature [C]', 'Process temperature [C]', 'Temp_difference [C]']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    data = request.form

    # Read form data and convert to DataFrame
    input_data = pd.DataFrame({
        'Type': [data['Type']],
        'Air temperature [C]': [float(data['Air_temp_C'])],
        'Process temperature [C]': [float(data['Process_temp_C'])],
        'Rotational speed [rpm]': [float(data['Rot_speed'])],
        'Torque [Nm]': [float(data['Torque'])],
        'Tool wear [min]': [float(data['Tool_wear'])]
    })

    # Add the missing 'Temp_difference [C]' feature
    input_data['Temp_difference [C]'] = input_data['Process temperature [C]'] - input_data['Air temperature [C]']

    # Ensure the DataFrame has the correct columns in the correct order
    input_data = input_data[feature_columns]

    # Encode categorical features (e.g., 'Type')
    input_data['Type'] = label_encoder.transform(input_data['Type'])

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Get prediction probabilities
    prediction = model.predict(input_data_scaled)
    probabilities = model.predict_proba(input_data_scaled)

    # Map failure type probabilities with class names
    failure_types = model.classes_

    # Check if prediction is valid and convert it to an integer index
    if len(prediction) > 0 and isinstance(prediction[0], (str, np.str_)):
        most_likely_failure = prediction[0]
        most_likely_prob = max(probabilities[0]) * 100  # Convert to percentage
    else:
        return jsonify({'error': 'Invalid prediction value'})

    # Convert probabilities to a dictionary with percentage values
    detailed_probabilities = {failure_types[i]: prob * 100 for i, prob in enumerate(probabilities[0])}

    # Return the prediction result along with probabilities
    return jsonify({
        'prediction': most_likely_failure,
        'probability': f"{most_likely_prob:.2f}%",
        'detailed_probabilities': detailed_probabilities
    })

if __name__ == '__main__':
    app.run(debug=True)

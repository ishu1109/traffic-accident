import pandas as pd
from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained pipeline model (this should include both preprocessor and model)
model = joblib.load('traffic_accident_pipeline.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        input_data = {
            'Weather': request.form['weather'],
            'Road_Type': request.form['road_type'],
            'Time_of_Day': request.form['time_of_day'],
            'Traffic_Density': float(request.form['traffic_density']),
            'Speed_Limit': float(request.form['speed_limit']),
            'Number_of_Vehicles': float(request.form['num_vehicles']),
            'Driver_Alcohol': float(request.form['driver_alcohol']),
            'Accident_Severity': request.form['accident_severity'],
            'Road_Condition': request.form['road_condition'],
            'Vehicle_Type': request.form['vehicle_type'],
            'Driver_Age': float(request.form['driver_age']),
            'Driver_Experience': float(request.form['driver_experience']),
            'Road_Light_Condition': request.form['road_light_condition']
        }

        # Convert to DataFrame (important to match exact column names as trained pipeline)
        input_df = pd.DataFrame([input_data])

        # Debug print to confirm data coming from form
        print("\nInput DataFrame being passed to model:\n", input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Interpret result
        result = "Accident Likely" if prediction == 1 else "No Accident Likely"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

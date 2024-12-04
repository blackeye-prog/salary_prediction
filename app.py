from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models and scaler from pickle files
scaler = pickle.load(open("model/scaler.pkl", "rb"))
model = pickle.load(open("model/regression.pkl", "rb"))

@app.route('/api/data', methods=['GET', 'POST'])
def predictSalary():
    try:
        # Handle GET request and retrieve data directly
        if request.method == 'GET':
            years_of_experience = request.args.get('experience', type=float)
            education_level = request.args.get('education', type=float)
            print(education_level, years_of_experience)
        # Handle POST request with form data
        elif request.method == 'POST':
            years_of_experience = request.form.get('experience', type=float)
            education_level = request.form.get('education', type=float)
            print(education_level, years_of_experience)
        else:
            return jsonify({"error": "Unsupported HTTP method"}), 405  # Method Not Allowed

        # Validate input data
        if education_level is None or years_of_experience is None:
            return jsonify({"error": "Missing required parameters"}), 400  # Bad Request

        # For simplicity, use the raw education_level value without converting it to a numeric value
        # If needed, you can process the `education_level` string directly in your model
        # Assuming your model already handles the string (e.g., "Bachelor's", "Master's", etc.)

        # Standardize the input data using the scaler
        new_data = scaler.transform([[education_level, years_of_experience]])

        # Perform prediction using the trained model
        prediction = model.predict(new_data)

        # Return the prediction as a response
        result = int(prediction[0])  # Convert the prediction to integer
        return jsonify({"result": result}), 200  # Return prediction result in JSON format

    except Exception as e:
        # Handle any errors (e.g., model issues, missing files, etc.)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500  # Internal Server Error

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(port=5000)

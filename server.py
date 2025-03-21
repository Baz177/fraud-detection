from flask import Flask, render_template, request
import joblib
import numpy as np
import csv
from datetime import datetime
import pandas as pd

model = joblib.load('fraud_model.pkl')

LOG_FILE = 'transaction_log.csv' # Log file for transactions

# Initialize the CSV file with headers if it doesn’t exist
def init_log_file():
    try:
        with open(LOG_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'distance_from_home', 'distance_from_last_transaction', 
                             'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 
                             'used_pin_number', 'online_order', 'prediction'])
    except FileExistsError:
        pass  # File already exists, no need to initialize

# Call this once when the app starts
init_log_file()

app = Flask(__name__)
@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/check_transaction', methods = ['POST']) 
def check_transaction():
    # Get the data from the form
    data = request.form.to_dict()
    print(data)

    # Define the expected field names from index.html
    field_names = [
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
        "repeat_retailer",
        "used_chip",
        "used_pin_number",
        "online_order"
    ]

    # Extract and convert inputs to floats
    try:
        input_data = [float(data[field]) for field in field_names]
    except (KeyError, ValueError) as e:
        return f"Error: Invalid or missing input data ({str(e)})", 400
    
    # Reshape the data
    input_array = np.array(input_data).reshape(1, -1)

    # Make the prediction
    prediction = int(model.predict(input_array)[0])
    print(f"Prediction: {prediction}")
    status = "Fraudulent" if prediction == 0 else "Legitimate"

    # Log the transaction to CSV
    with open(LOG_FILE, 'a', newline='') as f:  # 'a' for append mode
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + input_data + [prediction])

    # Check if the log file is full and prompt for retraining
    file_size = pd.read_csv(LOG_FILE).shape[0]
    if file_size > 1000:
        print("Log file has a full batch. Model would need to be retrained")

    # Render the template
    return render_template(
        "transaction.html",
        status = status,
        title = "Transaction Status",
        inputs={field: data[field] for field in field_names}
    )


if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8000)

from flask import Flask, render_template, request
import joblib
import numpy as np
import csv
from datetime import datetime
import pandas as pd
from waitress import serve
import os
from pyngrok import ngrok, conf
import getpass
import threading
from dotenv import load_dotenv

load_dotenv() 

ngrok_token = os.getenv("NGROK_AUTH_TOKEN")

model = joblib.load('fraud_model.pkl')

LOG_FILE = 'transaction_log.csv' # Log file for transactions

# Initialize the CSV file with headers if it doesn’t exist
def init_log_file():
    try:
        with open(LOG_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'distance_from_home', 'distance_from_last_transaction', 
                             'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 
                             'used_pin_number', 'online_order', 'prediction', 'verification'])
    except FileExistsError:
        pass  # File already exists, no need to initialize

# Call this once when the app starts
init_log_file()

field_names = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order"
] 

app = Flask(__name__)

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = getpass.getpass(ngrok_token)

port = 5000
os.environ["FLASK_DEBUG"] = "development"

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/check_transaction', methods=['POST'])
def check_transaction():
    # Get the data from the form
    data = request.form.to_dict()
    print("Form data:", data)

    # Validate inputs
    try:
        inputs = {field: float(data[field]) for field in field_names}
    except (KeyError, ValueError) as e:
        return f"Error: Invalid or missing input data ({str(e)})", 400

    # Render verification page (no prediction or logging here)
    return render_template(
        "confirmation.html",
        title="Confirm Transaction",
        inputs=inputs
    )

@app.route('/predict_transaction', methods=['POST'])
def predict_transaction():
    # Get the verified data from the form
    data = request.form.to_dict()
    print("Confirm data:", data)
   
    # Extract and convert inputs to floats
    try:
        input_data = [float(data[field]) for field in field_names]
    except (KeyError, ValueError) as e:
        return f"Error: Invalid or missing input data ({str(e)})", 400

    # Make prediction with the local model
    input_array = np.array(input_data).reshape(1, -1)
    prediction = int(model.predict(input_array)[0])  
    status = "Fraudulent" if prediction == 1 else "Legitimate"
    print(f"Prediction: {prediction}, {status}")

    if status == "Legitimate":
        verification_result = "Verified"

        # Log the transaction with verification
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + input_data + [prediction, verification_result])

        # Check log file size
        file_size = pd.read_csv(LOG_FILE).shape[0]
        if file_size > 1000:
            for _ in range(25):
                print('*******'*10)
                print("Log file has a full batch. Model would need to be retrained")
        
        # Final result       
        return render_template("transaction.html", title="Transaction Result", 
                          status=status, verification_result=verification_result, inputs=data)
    
    else:
        return render_template("verification.html", title="Verify Prediction",
                          status=status, inputs=data, prediction=prediction)

    
@app.route('/verify_transaction', methods=['POST'])
def verify_transaction():
    data = request.form.to_dict()
    print("Verification data:", data)
    try:
        input_data = [float(data[field]) for field in field_names]
        prediction = int(data['prediction'])
        verified_outcome = data.get('verified_outcome') 
    except (KeyError, ValueError) as e:
        return f"Error: Invalid or missing input data ({str(e)})", 400
    
    # Final result
    status = "Fraudulent" if prediction == 1 else "Legitimate"
    verification_result = "Verified" if verified_outcome == "yes" else "Not Verified"

    # Log the transaction with verification
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + input_data + 
                        [prediction, verification_result])

    # Check log file size
    file_size = pd.read_csv(LOG_FILE).shape[0]
    if file_size > 1000:
        for _ in range(25):
            print('*******'*10)
            print("Log file has a full batch. Model would need to be retrained")

    return render_template("transaction.html", title="Transaction Result", 
                          status=status, verification_result=verification_result, inputs=data)


if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

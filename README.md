# Fraud Detection App
##### A simple web application built with Flask to predict whether a transaction is fraudulent or legitimate based on user-provided input data. The app uses a pre-trained machine learning model, logs all transactions to a CSV file, and provides a basic user interface for input and result display.

### Features
* Input Form: Enter transaction details (e.g., distances, purchase ratio, binary flags).
* Prediction: Uses a pre-trained model (fraud_model.pkl) to classify transactions as "Fraudulent" or "Legitimate".
* Transaction Logging: Saves all inputs and results to transaction_log.csv with timestamps.
* Simple UI: Basic styling with a form and result page, including a "Return to Form" link.

### Usage
  ##### Enter values for 7 fields:
    * Distance from Home: Distance in kilometers (e.g., 57.87).
    * Distance from Last Transaction: Distance in kilometers (e.g., 0.31).
    * Ratio to Median Purchase Price: A ratio (e.g., 1.94).
    * Repeat Retailer, Used Chip Card, Used Pin Number, Online Order: Binary (0 or 1).
 ##### Click "Submit".

### Application Structure 

* fraud_app/
* ├── .fraud/                  # Virtual environment
* ├── static/
* │   └── styles/
* │       └── style.css        # Basic CSS styling
* ├── templates/
* │   ├── index.html           # Form page
* │   └── transaction.html     # Result page
* ├── fraud_model.pkl          # Pre-trained model
* ├── transaction_log.csv      # Log file (created on first run)
* ├── server.py                # Flask app
* └── README.md                # This file

[Fraud-Detection-Application](https://fraud-detection-knd8.onrender.com)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import random
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImPipline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import time
from datetime import timedelta
import joblib

# Load the data
trans_data = pd.read_csv(r'C:\Users\bkt29\OneDrive\Desktop\MLE_AI\datasets\card_transdata.csv')

#Function to remove outliers
def rem_outliers(df, col):
    """function to remove outliers"""

    #percentiles
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    #IQR
    IQR = Q3 - Q1

    #Define Lower and Upper bounds
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    df[col] = df[col].clip(lower = lower_bound, upper = upper_bound)

    return df

#Remove outliers from the data 
df_2 = rem_outliers(trans_data, 'distace_from_home')
df_1 = rem_outliers(trans_data, 'distance_from_last_transaction')
df = rem_outliers(trans_data, 'ratio_to_median_purchase_price')

def process_data(df, batch_size):

    # Balancing and Splitting data
    X = df.drop(columns = 'fraud', axis = 1)
    y = df['fraud']
    smote = SMOTE(sampling_strategy = 'auto', random_state = 10, k_neighbors = 4)
    X_oversampled, y_oversampled = smote.fit_resample(X, y)
    
    # Splitting data between train and tests sets 
    X_train, X_test, y_train, y_test = train_test_split(X_oversampled.values, y_oversampled.values, test_size=0.2, random_state=10, stratify = y_oversampled)
    
    # Streamlining data for faster execution. 
    global pipe
    pipe = Pipeline([('scalar', StandardScaler()), 
                    ('pca', PCA(n_components = .90, random_state = 10)), # Apply PCA to only carry important featurers
                    ('random_forest_classifier', RandomForestClassifier(max_depth = 20, 
                                                                        min_samples_split = 12, 
                                                                        n_estimators = 150, 
                                                                        n_jobs = -1))]) # Applying Hypertuned Classifier

    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))  # Calculate number of batches

    # Training batches 
    print('Model Training has begun ...Please wait')
    print('. . . . ....') 
    accuracy_scores = []
    training_start_time = time.time()
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)  # Handle last batch
        batch_X = X_train[start:end]
        batch_y = y_train[start:end]
        pipe.fit(batch_X, batch_y)

        # Recording accuracy
        accuracy = accuracy_score(batch_y, pipe.predict(batch_X))
        accuracy_scores.append(accuracy)
        
    # Time Record
    elapsed_time = time.time() - training_start_time
    formatted_time = str(round(timedelta(seconds=elapsed_time).total_seconds()/60,2))
    print(f"Training {num_batches} batches took: {formatted_time} mins ")
        
    # Checking accuracy of Model
    sns.histplot(x = accuracy_scores, bins = 25, kde = True, legend = False, color="skyblue")
    plt.title('Accuracy distribution over batches') 
    plt.xlabel('Acuracy')
    plt.show()
    cm = confusion_matrix(y_test, pipe.predict(X_test))
    
    sns.heatmap(cm, annot = True, fmt=  'g', cmap = 'Blues')
    plt.show()
    return classification_report(y_test, pipe.predict(X_test))

print(process_data(trans_data, 1000))

joblib.dump(pipe, 'fraud_model.pkl')
print("Model trained and saved as 'fraud_model.pkl'")

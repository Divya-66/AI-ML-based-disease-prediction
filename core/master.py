import pandas as pd
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Import the actual diabetes model
from diabetes.diabetes import predict_diabetes

# Temporary placeholder for heart module
class DummyHeart:
    @staticmethod
    def predict(df):
        return "Flow test successful - Heart model placeholder"

def process_csv(file):
    # Read the uploaded CSV file
    df = pd.read_csv(file)
    
    # Determine the model to use based on column names
    if any(col.lower() in ['glucose', 'glucose_value', 'diabetes'] for col in df.columns):
        return predict_diabetes(df)  # Use actual diabetes model
    elif any(col.lower() in ['heart_rate', 'ecg'] for col in df.columns):
        return DummyHeart.predict(df)  # Use dummy heart model
    else:
        return "Unknown data type. Please upload data with relevant columns (e.g., glucose_value, steps, sleep_hours, calories)."

# Placeholder for future expansion
def process_input(user_input):
    return "Input processing not implemented yet."
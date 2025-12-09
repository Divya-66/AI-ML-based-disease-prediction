import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os

# Global variables for model components (loaded when needed)
df_train = None
daily_df_train = None
features = ['glucose_value', 'steps', 'sleep_hours', 'calories']
data_train = None
adj = None
model = None
device = None

def load_training_data():
    """Load training data and initialize model components"""
    global df_train, daily_df_train, data_train, adj, model, device
    
    if model is not None:
        return  # Already loaded
    
    try:
        # Load training data for adjacency
        df_train = pd.read_csv('patient_combined_data (7).csv')
        daily_df_train = df_train.groupby(['patient_id', 'day']).agg({
            'glucose_value': 'mean',
            'steps': 'sum',
            'sleep_hours': 'mean',
            'calories': 'sum',
            'is_diabetic': 'first'
        }).reset_index()
        data_train = daily_df_train[features].values

        # Compute adjacency matrix
        corr = np.corrcoef(data_train.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        A = corr + np.eye(corr.shape[0])
        D = np.diag(1.0 / np.sqrt(np.clip(A.sum(axis=1), 1e-8, None)))
        A_hat = D @ A @ D
        adj = torch.from_numpy(A_hat).float().to(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        feature_size = len(features)
        hidden_size = 64
        model = GNNLSTMODE(feature_size=feature_size, hidden_size=hidden_size, adj=adj).to(device)
        model_path = 'model.pth'  # Load from same folder
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required training data files not found: {e}. Please ensure 'patient_combined_data (7).csv' and 'model.pth' are in the models/diabetes/ directory.")
    except Exception as e:
        raise Exception(f"Error loading training data: {e}")

# Custom scaler
def minmax_scale(arr, eps=1e-8):
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    scaled = (arr - mn) / (mx - mn + eps)
    return scaled, mn, mx

# Neural ODE component
class ODEFunc(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.noise_scale = 0.01

    def forward(self, t, y):
        noise = torch.randn_like(y) * self.noise_scale
        return self.net(y) + noise

# Model definition
class GNNLSTMODE(nn.Module):
    def __init__(self, feature_size, hidden_size, adj):
        super().__init__()
        self.adj = adj
        self.gcn = nn.Linear(feature_size, hidden_size)
        self.ode_func = ODEFunc(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, T, F = x.shape
        x = x.to(self.adj.device)
        gnn_seq = []
        for t in range(T):
            xt = x[:, t, :]
            xt = torch.matmul(xt, self.adj.T)
            ht = torch.relu(self.gcn(xt))
            gnn_seq.append(ht)
        gnn_seq = torch.stack(gnn_seq, dim=1)
        
        t_eval = torch.linspace(0, 1, T, device=self.adj.device)
        y0 = gnn_seq[:, 0, :]
        ode_sol = odeint(self.ode_func, y0, t_eval, method='rk4')
        ode_seq = ode_sol.permute(1, 0, 2)
        
        lstm_out, _ = self.lstm(ode_seq)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Model loading is now handled in load_training_data() function

# Forecast function with corrected trend extrapolation
def forecast_60_days(model, sample_sequence, adj, global_min, global_max, days=60, noise_scale=0.02):
    predictions = []
    current_seq = torch.from_numpy(sample_sequence).float().unsqueeze(0).to(device)
    current_values = sample_sequence[-1].copy()
    
    trends = np.diff(sample_sequence, axis=0)
    avg_trend = np.mean(trends, axis=0)
    
    for _ in range(days):
        with torch.no_grad():
            pred_glucose_scaled = model(current_seq)
        pred_glucose = pred_glucose_scaled.item() * (global_max[0] - global_min[0]) + global_min[0]
        
        # Add noise for realism
        pred_glucose += np.random.normal(0, noise_scale * global_max[0])
        predictions.append(pred_glucose)
        
        # Update features
        current_values[0] = pred_glucose_scaled.item()
        current_values[1:] += avg_trend[1:] + np.random.normal(0, 0.01, size=len(avg_trend[1:]))  # variability
        current_values = np.clip(current_values, 0, None)
        
        next_row = (current_values - global_min) / (global_max - global_min + 1e-8)
        next_row_tensor = torch.from_numpy(next_row).float().unsqueeze(0).unsqueeze(0).to(device)
        current_seq = torch.cat((current_seq[:, 1:, :], next_row_tensor), dim=1)
    
    return predictions

# Interface function for Flask app
def predict_diabetes(user_df):
    """
    Interface function for Flask app to predict diabetes risk from user data.
    
    Args:
        user_df: pandas DataFrame with user data containing columns like 'glucose_value', 'steps', 'sleep_hours', 'calories'
    
    Returns:
        str: Prediction result with risk assessment
    """
    try:
        # Check if required columns exist
        required_cols = ['glucose_value', 'steps', 'sleep_hours', 'calories']
        missing_cols = [col for col in required_cols if col not in user_df.columns]
        
        if missing_cols:
            return f"Error: Missing required columns: {missing_cols}. Please ensure your CSV has columns: {required_cols}"
        
        # Try to load training data and model
        try:
            load_training_data()
            
            # Process user data
            user_data = user_df[required_cols].values
            
            # Handle single row of data
            if len(user_data) == 1:
                # For single data point, create a sequence by repeating the data
                user_data = np.tile(user_data, (10, 1))
            elif len(user_data) < 10:
                # For less than 10 days, pad with the last available data
                last_row = user_data[-1:]
                padding = np.tile(last_row, (10 - len(user_data), 1))
                user_data = np.vstack([user_data, padding])
            else:
                # Take the last 10 days
                user_data = user_data[-10:]
            
            # Scale the data
            user_scaled, user_min, user_max = minmax_scale(user_data)
            
            # Make prediction
            time_steps = 10
            sample_input = user_scaled[-time_steps:]
            predictions = forecast_60_days(model, sample_input, adj, user_min, user_max, days=60)
            
            # Calculate risk metrics
            mean_pred_glucose = np.mean(predictions)
            risk_score = np.mean(np.array(predictions) > 250) * 100
            
            # Determine risk category
            if mean_pred_glucose > 126:
                risk_category = "High Risk (Diabetes)"
            elif mean_pred_glucose > 100:
                risk_category = "Medium Risk (Prediabetes)"
            else:
                risk_category = "Low Risk"
            
            # Format result
            result = f"""
Diabetes Risk Assessment:
- Mean Predicted Glucose: {mean_pred_glucose:.2f} mg/dL
- Risk Score: {risk_score:.2f}% (days with glucose > 250 mg/dL)
- Risk Category: {risk_category}
- Prediction based on next 60 days forecast
            """.strip()
            
            return result
            
        except FileNotFoundError:
            # Fallback: Simple analysis without trained model
            return predict_diabetes_simple(user_df)
        
    except Exception as e:
        required_cols = ['glucose_value', 'steps', 'sleep_hours', 'calories']
        return f"Error processing data: {str(e)}. Please ensure your CSV has the correct format with columns: {required_cols}"

def predict_diabetes_simple(user_df):
    """
    Simple diabetes prediction without trained model (fallback)
    """
    try:
        required_cols = ['glucose_value', 'steps', 'sleep_hours', 'calories']
        
        # Get basic statistics
        glucose_values = user_df['glucose_value'].values
        steps_values = user_df['steps'].values
        sleep_values = user_df['sleep_hours'].values
        calories_values = user_df['calories'].values
        
        # Calculate basic metrics
        avg_glucose = np.mean(glucose_values)
        max_glucose = np.max(glucose_values)
        avg_steps = np.mean(steps_values)
        avg_sleep = np.mean(sleep_values)
        avg_calories = np.mean(calories_values)
        
        # Simple risk assessment based on glucose levels
        if avg_glucose > 126 or max_glucose > 200:
            risk_category = "High Risk (Diabetes)"
            risk_score = 85.0
        elif avg_glucose > 100 or max_glucose > 140:
            risk_category = "Medium Risk (Prediabetes)"
            risk_score = 45.0
        else:
            risk_category = "Low Risk"
            risk_score = 15.0
        
        # Lifestyle factors
        lifestyle_score = 0
        if avg_steps < 5000:
            lifestyle_score += 20
        if avg_sleep < 6 or avg_sleep > 9:
            lifestyle_score += 15
        if avg_calories < 1500 or avg_calories > 3000:
            lifestyle_score += 10
            
        # Adjust risk score based on lifestyle
        risk_score = min(100, risk_score + lifestyle_score)
        
        # Generate graphs
        graph_files = generate_analysis_graphs(user_df, avg_glucose, risk_score, risk_category)
        
        result = f"""
Diabetes Risk Assessment (Simple Analysis):
- Average Glucose: {avg_glucose:.2f} mg/dL
- Maximum Glucose: {max_glucose:.2f} mg/dL
- Average Steps: {avg_steps:.0f} steps/day
- Average Sleep: {avg_sleep:.1f} hours/day
- Average Calories: {avg_calories:.0f} cal/day
- Risk Score: {risk_score:.1f}%
- Risk Category: {risk_category}
- Lifestyle Factors: {'Poor' if lifestyle_score > 20 else 'Good' if lifestyle_score < 10 else 'Moderate'}

Health Recommendations:
- {'⚠️ High glucose levels detected. Consider dietary changes and exercise.' if avg_glucose > 126 else '✅ Glucose levels are within normal range.'}
- {'⚠️ Low activity level. Aim for at least 5,000 steps daily.' if avg_steps < 5000 else '✅ Good activity level maintained.'}
- {'⚠️ Sleep pattern needs improvement. Target 7-9 hours nightly.' if avg_sleep < 6 or avg_sleep > 9 else '✅ Healthy sleep pattern maintained.'}
- {'⚠️ Calorie intake may need adjustment.' if avg_calories < 1500 or avg_calories > 3000 else '✅ Calorie intake is balanced.'}

Note: This is a basic analysis. For accurate predictions, please ensure training data files are available.

Graphs Generated: {', '.join(graph_files)}
        """.strip()
        
        return result
        
    except Exception as e:
        return f"Error in simple analysis: {str(e)}"

def generate_analysis_graphs(user_df, avg_glucose, risk_score, risk_category):
    """
    Generate analysis graphs and save them as static files
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import os
        
        # Create static directory if it doesn't exist
        static_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dashboard', 'static')
        os.makedirs(static_dir, exist_ok=True)
        
        graph_files = []
        
        # 1. Glucose Trend Chart
        plt.figure(figsize=(10, 6))
        plt.plot(user_df.index, user_df['glucose_value'], marker='o', linewidth=2, markersize=6)
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Normal (100 mg/dL)')
        plt.axhline(y=126, color='orange', linestyle='--', alpha=0.7, label='Prediabetes (126 mg/dL)')
        plt.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='High Risk (200 mg/dL)')
        plt.xlabel('Data Points')
        plt.ylabel('Glucose Level (mg/dL)')
        plt.title('Glucose Level Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        glucose_file = os.path.join(static_dir, 'glucose_trend.png')
        plt.savefig(glucose_file, dpi=300, bbox_inches='tight')
        plt.close()
        graph_files.append('glucose_trend.png')
        
        # 2. Lifestyle Metrics Chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Steps
        ax1.bar(['Average Steps'], [user_df['steps'].mean()], color='blue', alpha=0.7)
        ax1.axhline(y=5000, color='red', linestyle='--', alpha=0.7, label='Minimum Recommended')
        ax1.set_ylabel('Steps')
        ax1.set_title('Daily Steps')
        ax1.legend()
        
        # Sleep
        ax2.bar(['Average Sleep'], [user_df['sleep_hours'].mean()], color='green', alpha=0.7)
        ax2.axhline(y=7, color='red', linestyle='--', alpha=0.7, label='Recommended (7-9h)')
        ax2.axhline(y=9, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Hours')
        ax2.set_title('Daily Sleep')
        ax2.legend()
        
        # Calories
        ax3.bar(['Average Calories'], [user_df['calories'].mean()], color='orange', alpha=0.7)
        ax3.axhline(y=1500, color='red', linestyle='--', alpha=0.7, label='Minimum')
        ax3.axhline(y=2500, color='red', linestyle='--', alpha=0.7, label='Maximum')
        ax3.set_ylabel('Calories')
        ax3.set_title('Daily Calories')
        ax3.legend()
        
        # Risk Score
        colors = ['green' if risk_score < 30 else 'orange' if risk_score < 70 else 'red']
        ax4.bar(['Risk Score'], [risk_score], color=colors, alpha=0.7)
        ax4.set_ylabel('Risk Score (%)')
        ax4.set_title(f'Diabetes Risk: {risk_category}')
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        lifestyle_file = os.path.join(static_dir, 'lifestyle_metrics.png')
        plt.savefig(lifestyle_file, dpi=300, bbox_inches='tight')
        plt.close()
        graph_files.append('lifestyle_metrics.png')
        
        # 3. Correlation Heatmap
        plt.figure(figsize=(8, 6))
        correlation_data = user_df[['glucose_value', 'steps', 'sleep_hours', 'calories']].corr()
        im = plt.imshow(correlation_data, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation')
        plt.xticks(range(len(correlation_data.columns)), correlation_data.columns, rotation=45)
        plt.yticks(range(len(correlation_data.columns)), correlation_data.columns)
        plt.title('Feature Correlation Matrix')
        
        # Add correlation values to the heatmap
        for i in range(len(correlation_data.columns)):
            for j in range(len(correlation_data.columns)):
                plt.text(j, i, f'{correlation_data.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontweight='bold')
        
        plt.tight_layout()
        correlation_file = os.path.join(static_dir, 'correlation_heatmap.png')
        plt.savefig(correlation_file, dpi=300, bbox_inches='tight')
        plt.close()
        graph_files.append('correlation_heatmap.png')
        
        # 4. Risk Assessment Pie Chart
        plt.figure(figsize=(8, 8))
        risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_colors = ['green', 'orange', 'red']
        
        if risk_score < 30:
            risk_distribution = [100, 0, 0]
        elif risk_score < 70:
            risk_distribution = [30, 70, 0]
        else:
            risk_distribution = [10, 30, 60]
        
        plt.pie(risk_distribution, labels=risk_levels, colors=risk_colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Diabetes Risk Assessment\nCurrent Risk: {risk_score:.1f}%')
        plt.axis('equal')
        
        risk_file = os.path.join(static_dir, 'risk_assessment.png')
        plt.savefig(risk_file, dpi=300, bbox_inches='tight')
        plt.close()
        graph_files.append('risk_assessment.png')
        
        return graph_files
        
    except Exception as e:
        print(f"Error generating graphs: {e}")
        return []

# Only run the full analysis if this script is run directly (not imported)
if __name__ == "__main__":
    # Load sample data
    try:
        sample_df = pd.read_excel('sample (2).xlsx')  # Load from same folder
        sample_daily = sample_df.groupby(['patient_id', 'day']).agg({
            'glucose_value': 'mean',
            'steps': 'sum',
            'sleep_hours': 'mean',
            'calories': 'sum',
            'is_diabetic': 'first'
        }).reset_index()
    except Exception as e:
        print(f"Error loading sample data: {e}")
        print("Using last 10 days from training data as fallback...")
        sample_daily = daily_df_train[daily_df_train['day'].isin(range(51, 61))]  # Days 51–60

    # Predict for all patients
    time_steps = 10
    results = []
    for pid in sample_daily['patient_id'].unique():
        patient_data = sample_daily[sample_daily['patient_id'] == pid][features].values
        if len(patient_data) < time_steps:
            print(f"Skipping patient {pid}: insufficient data ({len(patient_data)} days)")
            continue
        sample_input = patient_data[-time_steps:]  # Last 10 days
        sample_scaled, patient_min, patient_max = minmax_scale(sample_input)
        
        predictions = forecast_60_days(model, sample_scaled, adj, patient_min, patient_max)
        
        # Compute metrics
        mean_pred_glucose = np.mean(predictions)
        risk_score = np.mean(np.array(predictions) > 250) * 100
        is_diabetic = sample_daily[sample_daily['patient_id'] == pid]['is_diabetic'].iloc[0]
        
        risk_category = "High Risk (Diabetes)" if mean_pred_glucose > 126 else "Medium Risk (Prediabetes)" if mean_pred_glucose > 100 else "Low Risk"
        
        results.append({
            'patient_id': pid,
            'mean_glucose': mean_pred_glucose,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'is_diabetic': is_diabetic,
            'predictions': predictions
        })
        
        # Plot for sample patients
        if pid in [1, 40]:
            plt.figure(figsize=(10, 5))
            plt.plot(predictions, label=f'Patient {pid} Predicted Glucose')
            plt.axhline(y=250, color='r', linestyle='--', label='High Glucose Threshold (250)')
            plt.xlabel('Future Days (61-120)')
            plt.ylabel('Glucose Level (mg/dL)')
            plt.title(f'Patient {pid} Glucose Forecast (Diabetic: {is_diabetic})')
            plt.legend()
            plt.show()

    # Analytical summary
    results_df = pd.DataFrame([{
        'Patient ID': r['patient_id'],
        'Mean Predicted Glucose (mg/dL)': round(r['mean_glucose'], 2),
        'Risk Score (>250 mg/dL, %)': round(r['risk_score'], 2),
        'Risk Category': r['risk_category'],
        'Is Diabetic': r['is_diabetic']
    } for r in results])

    print("\nPrediction Summary for Next 60 Days:")
    print(results_df.to_string(index=False))

    # Risk score distribution
    plt.figure(figsize=(10, 5))
    plt.hist([r['risk_score'] for r in results], bins=20, edgecolor='k')
    plt.xlabel('Risk Score (% of Days with Glucose > 250 mg/dL)')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Risk Scores Across Patients')
    plt.show()

    # Feature correlation heatmap
    corr_matrix = pd.DataFrame(corr, index=features, columns=features)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(np.arange(len(features)), features, rotation=45)
    plt.yticks(np.arange(len(features)), features)
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Summary analysis
    high_risk = results_df[results_df['Risk Category'] == 'High Risk (Diabetes)']
    print(f"\nAnalysis:")
    print(f"- {len(high_risk)} patients are at high risk (mean glucose > 126 mg/dL).")
    print(f"- Highest risk score: Patient {results_df.loc[results_df['Risk Score (>250 mg/dL, %)'].idxmax(), 'Patient ID']} "
          f"({results_df['Risk Score (>250 mg/dL, %)'].max():.2f}%).")
    print(f"- Diabetic patients: {results_df['Is Diabetic'].sum()} out of {len(results_df)}.")
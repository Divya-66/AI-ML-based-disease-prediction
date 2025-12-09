from flask import Flask, render_template, request, send_from_directory
import os
from core import master

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    prediction = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                prediction = master.process_csv(file)  # Pass the file to master.py
    return render_template('index.html', prediction=prediction)

@app.route('/static/<filename>')
def static_files(filename):
    """Serve static files (graphs) from the dashboard/static directory"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
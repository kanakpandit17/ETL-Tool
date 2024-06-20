from flask import Flask, render_template, request, send_file,jsonify, abort, request, redirect
from flask import Response
from confluent_kafka import Producer
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from flask import Flask, render_template, request, jsonify
import csv
import time
import math
import json
import numpy as np
import gzip
from flask import Response
from flask import send_from_directory
from flask import send_file
import time
import gzip
import pandas as pd
from io import BytesIO
import base64
import sqlite3
import zlib
import zipfile
import hashlib
from collections import Counter, defaultdict
import heapq
import pickle
import os
import pycaret
from pycaret.classification import *
import csv
from io import TextIOWrapper
from text_prettifier import TextPrettifier
import nltk
nltk.download('stopwords')



# Define the directory to save converted files
CONVERTED_FILES_DIR = 'c:/Users/kanakpandit17/Downloads/converted'

# Ensure the directory exists before saving any files
if not os.path.exists(CONVERTED_FILES_DIR):
    os.makedirs(CONVERTED_FILES_DIR)

DOWNLOADS_DIR = 'c:/Users/kanakpandit17/Downloads'


app = Flask(__name__)



# Function to setup PyCaret environment and train model
def train_model(train_data):
    s = setup(train_data, target='target', session_id=123)
    best = compare_models()
    evaluate_model(best)
    save_model(best, 'best_pipeline')
    return best

# Function to make predictions on test data
def make_predictions(model, test_data):
    predictions = predict_model(model, data=test_data)
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        # Check if files are uploaded
        if 'train_file' not in request.files or 'test_file' not in request.files:
            return "No file part"

        # Get uploaded files
        train_file = request.files['train_file']
        test_file = request.files['test_file']

        # Read the uploaded CSV files
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Train model
        model = train_model(train_data)

        # Make predictions on test data
        predictions = make_predictions(model, test_data)

        # Save predictions to CSV
        predictions.to_csv(CONVERTED_FILES_DIR + '/predicted_test_data.csv', index=False)

        # Return the template with predictions
        return render_template('predictor.html', predictions=predictions.to_html())

    # Render predictor.html for GET requests
    return render_template('predictor.html')


@app.route('/data_compression', methods=['GET', 'POST'])
def data_compression():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"

        # Read the file into memory
        file_content = file.read()

        # Determine the file extension
        file_extension = os.path.splitext(file.filename)[1].lower()

        # Compress the file based on its extension
        if file_extension == '.csv':
            # Read the CSV content into a DataFrame
            content = pd.read_csv(BytesIO(file_content))

            # Convert the DataFrame to a bytes-like object (in this case, a CSV string)
            csv_content = content.to_csv(index=False).encode('utf-8')

            # Write the compressed data to a gzip file
            with gzip.open(os.path.join(CONVERTED_FILES_DIR, 'compressed_file.csv.gz'), 'wb') as f:
                f.write(csv_content)
        elif file_extension == '.json':
            # Load JSON data from the uploaded file
            json_data = json.loads(file_content.decode('utf-8'))

            # Compress the JSON data
            compressed_json = zlib.compress(json.dumps(json_data).encode('utf-8'))

            # Write the compressed JSON data to a gzip file
            with gzip.open(os.path.join(CONVERTED_FILES_DIR, 'compressed_file.json.gz'), 'wb') as f:
                f.write(compressed_json)
        else:
            return "Unsupported file format"

        # Return a message indicating successful compression
        return render_template('data_compression.html', success=True)

    # Render the data compression template for GET requests
    return render_template('data_compression.html', success=False)


# Function to analyze CSV file
def analyze_csv(file):
    # Read the CSV file
    df = pd.read_csv(file)

    # Perform analysis
    analysis_results = {
        'missing_values': df.isnull().sum().to_dict(),
        'rows_count': len(df),
        'columns_count': len(df.columns),
        'data_types': df.dtypes.to_dict(),
        # Add more analysis as needed
    }

    return analysis_results

# Function to generate graphs based on the uploaded CSV file
def generate_graphs(file):
    # Read the CSV file
    df = pd.read_csv(file)

    # Generate graphs
    graphs = []

    # Example: Create a histogram for each numerical column
    for column in df.select_dtypes(include='number').columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df[column], bins=20)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Convert the plot to base64 encoding
        image_uri = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        # Append the base64 encoded image URI and column name to the list of graphs
        graphs.append({'image': f'data:image/png;base64,{image_uri}', 'column': column})

    # Example: Create a pie chart for each categorical column
    for column in df.select_dtypes(include='object').columns:
        counts = df[column].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title(f'Pie Chart of {column}')
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Convert the plot to base64 encoding
        image_uri = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        # Append the base64 encoded image URI and column name to the list of graphs
        graphs.append({'image': f'data:image/png;base64,{image_uri}', 'column': column})

    # Example: Create a line plot for each pair of numerical columns
    numerical_columns = df.select_dtypes(include='number').columns
    for i, column1 in enumerate(numerical_columns):
        for column2 in numerical_columns[i+1:]:
            plt.figure(figsize=(8, 6))
            plt.plot(df[column1], df[column2], 'o')
            plt.title(f'Line Plot of {column1} vs {column2}')
            plt.xlabel(column1)
            plt.ylabel(column2)
            
            # Save the plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Convert the plot to base64 encoding
            image_uri = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()

            # Append the base64 encoded image URI and column names to the list of graphs
            graphs.append({'image': f'data:image/png;base64,{image_uri}', 'column': f'{column1} vs {column2}'})

    return graphs



# Route for the Data Insights page
@app.route('/data_insights', methods=['GET', 'POST'])
def data_insights():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"

        # Generate graphs based on the uploaded CSV file
        graphs = generate_graphs(file)

        # Render the data insights template with the generated graphs
        return render_template('data_insights.html', graphs=graphs)

    # Render the data insights template for GET requests
    return render_template('data_insights.html')

# Route for the Data Profiler page
@app.route('/data_profiler', methods=['GET', 'POST'])
def data_profiler():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"

        # Analyze the uploaded CSV file
        analysis_results = analyze_csv(file)

        # Render the data profiler template with the analysis results
        return render_template('data_profiler.html', analysis_results=analysis_results)

    # Render the data profiler template for GET requests
    return render_template('data_profiler.html')


@app.route('/data_stream_monitor', methods=['GET', 'POST'])
def data_stream_monitor():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Read the uploaded CSV file into a list of rows
        data = file.read().decode('utf-8').splitlines()

        # Initialize counter for data changes
        changes_count = 0

        # Initialize previous row and hash value
        prev_row = None
        prev_hash = None

        # List to store row data
        row_data_list = []

        # Iterate over each row in the dataset
        for row in data:
            # Calculate hash for the current row
            current_hash = calculate_hash(row)

            # If previous hash exists and it's different from the current hash, increment changes_count
            if prev_hash is not None and prev_hash != current_hash:
                changes_count += 1

            # Display row details
            response_data = display_row_data(row, changes_count)

            # Append row data to the list
            row_data_list.append(response_data)

            # Set current row and hash as previous for the next iteration
            prev_row = row
            prev_hash = current_hash

            # Sleep for a short time to simulate processing delay
            time.sleep(1)

        # Return all row data as a JSON response
        return jsonify(row_data_list)

    return render_template('data_stream_monitor.html')
    

def calculate_hash(data):
    # Custom hash function implementation
    # Concatenate all values and calculate a simple hash
    concatenated_data = ''.join(map(str, data))
    hash_value = 0
    
    for char in concatenated_data:
        hash_value = (31 * hash_value + ord(char)) & 0xFFFFFFFF
    
    return hash_value



def display_row_data(row, changes_count):
    # Display row details
    print("Changes:", changes_count)
    if changes_count == 0:
        print("Columns changed: No columns changed")
    else:
        print("Columns changed:", row)
    print("Log time:", time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))

    # Prepare the response data
    response_data = {
        'changes_count': changes_count,
        'columns_changed': row if changes_count > 0 else [],  # Only send changed columns if there are changes
        'log_time': time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
    }


    return response_data


@app.route('/data_format_converter', methods=['GET', 'POST'])
def data_format_converter():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"
        
        # Check if the file is a CSV file
        if file.filename.endswith('.csv'):
            # Convert the uploaded CSV file to JSON format
            json_file_path = convert_to_json(file)

            # Check if conversion was successful
            if json_file_path is None:
                return "Failed to convert file to JSON format"
            
            # Return the JSON file for download
            return send_file(json_file_path, as_attachment=True)

        elif file.filename.endswith('.json'):
            # Convert the uploaded JSON file to CSV format
            csv_file_path = convert_to_csv(file)

            # Check if conversion was successful
            if csv_file_path is None:
                return "Failed to convert file to CSV format"
            
            # Return the CSV file for download
            return send_file(csv_file_path, as_attachment=True)

        else:
            return "Unsupported file format"

    # Render the data format converter template for GET requests
    return render_template('data_format_converter.html')

def convert_to_json(file):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file, encoding='latin1')  # Specify a different encoding, such as 'latin1'

        # Convert DataFrame to JSON format
        json_file_path = os.path.join(CONVERTED_FILES_DIR, 'output.json')
        df.to_json(json_file_path, orient='records')
        return json_file_path
    except Exception as e:
        print(f"Error converting file to JSON: {e}")
        return None

def convert_to_csv(file):
    try:
        # Load JSON data from the uploaded file
        json_data = json.load(file)
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame(json_data)
        
        # Save DataFrame to CSV
        csv_file_path = os.path.join(CONVERTED_FILES_DIR, 'output.csv')
        df.to_csv(csv_file_path, index=False)
        
        return csv_file_path

    except Exception as e:
        print(f"Error converting JSON to CSV: {e}")
        return None

@app.route('/automatic_preprocess', methods=['GET', 'POST'])
def automatic_preprocess():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"
        
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)

        # Initialize TextPrettifier object
        prettifier = TextPrettifier()

        # Iterate over each column in the DataFrame
        for column in df.columns:
            # Convert non-string values to strings
            df[column] = df[column].astype(str)
            # Apply text cleaning using TextPrettifier's methods
            df[column] = df[column].apply(prettifier.sigma_cleaner)

        # Save the cleaned DataFrame to a new CSV file
        cleaned_file_path = os.path.join(CONVERTED_FILES_DIR, 'cleaned_file.csv')
        df.to_csv(cleaned_file_path, index=False)

        # Render the automatic preprocess template with the cleaned file path
        return render_template('automatic_preprocess.html', cleaned_file_path=cleaned_file_path)

    # Render the automatic preprocess template for GET requests
    return render_template('automatic_preprocess.html')

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/download_compressed')
# def download_compressed():
#     # Specify the path to the original file
#     original_file_path = '/Downloads/original_file.csv'

#     # Specify the path for the compressed file
#     compressed_file_path = 'compressed_files/compressed_file.zip'

#     # Create a zip file and add the original file to it
#     with zipfile.ZipFile(compressed_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         zipf.write(original_file_path, os.path.basename(original_file_path))

#     # Return the compressed file for download
#     return send_file(compressed_file_path, as_attachment=True)

@app.route('/C:/Users/kanakpandit17/Downloads')
def download_converted(filename):
    converted_files_dir = 'converted_files'  # Change this to the correct directory
    full_file_path = os.path.join(converted_files_dir, filename)
    
    # Check if the file exists
    if not os.path.exists(full_file_path):
        # Log the error
        app.logger.error(f"File not found: {full_file_path}")
        # Return an error response
        abort(404)

    # Attempt to send the file
    try:
        return send_file(full_file_path, as_attachment=True)
    except Exception as e:
        # Log the error
        app.logger.error(f"Error sending file: {e}")
        # Return an error response
        abort(500)



# @app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
# def download(filename):
#     print(app.root_path)
#     full_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
#     print(full_path)
#     return send_from_directory(full_path, filename)

# @app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
# def download(filename):
#     uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
#     return send_from_directory(uploads, filename)

# @app.route('/download_converted/<path:filename>')
# def download_converted(filename):
#     return send_file(directory=CONVERTED_FILES_DIR, filename=filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

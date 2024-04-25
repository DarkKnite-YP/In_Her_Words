from flask import Flask, render_template, request, redirect, url_for
import csv
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save():
    try:
        text = request.form['text']
        hostility = request.form['hostility']
        option = request.form.get('option', '')  # Get selected option, if any

        # Validate input (you can add more validation as needed)
        if not text or not hostility:
            return 'Text and Hostility fields are required.', 400

        # Write data to a CSV file
        with open('user_data.csv', 'a', newline='') as csvfile:
            fieldnames = ['Text', 'Hostility', 'Option']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file is empty, write headers
            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow({'Text': text, 'Hostility': hostility, 'Option': option})

        return 'Your data has been recorded.'
    except Exception as e:
        return f'An error occurred: {str(e)}', 500

@app.route('/check_hostility')
def check_hostility():
    return render_template('check_hostility.html')

@app.route('/check_result', methods=['POST'])
def check_result():
    try:
        text = request.form['text']

        # Perform hostility check (you can customize this)
        is_hostile = bool(re.search(r'hate|fake|offensive|defamation', text, re.IGNORECASE))

        return render_template('check_result.html', text=text, is_hostile=is_hostile)
    except Exception as e:
        return f'An error occurred: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)

FROM flask import Flask, render_template, request
import csv

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/save', methods=['POST'])
def save():
    text = request.form['text']
    hostility = request.form['hostility']
    option = request.form.get('option', '')  # Get selected option, if any

    # Write data to a CSV file
    with open('user_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['Text', 'Hostility', 'Option']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is empty, write headers
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'Text': text, 'Hostility': hostility, 'Option': option})

    return 'Your data has been recorded.'


if __name__ == '__main__':
    app.run(debug=True)

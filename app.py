from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Sample datasets
datasets = {
    "dataset1": {
        "x": [1, 2, 3, 4, 5],
        "y": [10, 15, 13, 17, 21],
        "z": [5, 10, 7, 12, 15]
    },
    "dataset2": {
        "x": [1, 2, 3, 4, 5],
        "y": [5, 10, 7, 12, 15],
        "z": [10, 15, 13, 17, 21]
    },
    "dataset3": {
        "x": [1, 2, 3, 4, 5],
        "y": [20, 18, 22, 19, 25],
        "z": [15, 12, 18, 14, 20]
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_data/<dataset_id>')
def get_data(dataset_id):
    # Return the selected dataset as JSON
    data = datasets.get(dataset_id, {"x": [], "y": [], "z": []})
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
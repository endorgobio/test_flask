from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import os
import json
from pyomo.environ import *
from pyomo.opt import SolverFactory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

data = {
    'Dataset 1': {'x': [1, 2, 3, 4, 5], 'y': [10, 15, 7, 12, 17]},
    'Dataset 2': {'x': [1, 2, 3, 4, 5], 'y': [5, 10, 15, 10, 5]},
    'Dataset 3': {'x': [1, 2, 3, 4, 5], 'y': [3, 6, 9, 12, 15]}
}

def create_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], mode='lines+markers', name='User Data'))
    fig.update_layout(title='Plotly Line Graph', xaxis_title='X Axis', yaxis_title='Y Axis')
    return fig.to_json(fig)

def create_plot_from_dataset(dataset_name):
    df = pd.DataFrame(data[dataset_name])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines+markers', name=dataset_name))
    fig.update_layout(title='Plotly Line Graph', xaxis_title='X Axis', yaxis_title='Y Axis')
    return fig.to_json(fig)

@app.route('/')
def index():
    return render_template('index.html', datasets=list(data.keys()))

@app.route('/update_graph', methods=['POST'])
def update_graph():
    dataset_name = request.json.get('dataset')
    graph_json = create_plot_from_dataset(dataset_name)
    return jsonify(graph_json)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_json(filepath)
        graph_json = create_plot(df)
        return jsonify(graph_json)
    
@app.route('/run_model', methods=['POST'])
def run_model():
    try:
        data = request.get_json()
        file_data = data.get("file_data")  # Extract file content

        
        parsed_data = json.loads(file_data)  # Parse JSON file if needed
        values = parsed_data["values"]
        weights = parsed_data["weights"]
        capacity = parsed_data["capacity"]
        num_items = len(values)

        # Create Pyomo model
        model = ConcreteModel()

        # Define sets
        model.Items = Set(initialize = list(range(num_items)))
        print(model.Items)

        # Decision variables (Binary: 0 or 1)
        model.x = Var(model.Items, within=Binary)

        # Objective function: Maximize value
        model.obj = Objective(expr=sum(values[i] * model.x[i] for i in model.Items), sense=maximize)

        # Constraint: Total weight should not exceed capacity
        model.weight_constraint = Constraint(expr=sum(weights[i] * model.x[i] for i in model.Items) <= capacity)

        # # Solve using HiGHS solver
        # solver = SolverFactory("highs")
        # result = solver.solve(model)

        # # Extract results
        # selected_items = [i for i in model.items if model.x[i].value == 1]

        # # Return solution
        # return {
        #     "status": str(result.solver.status),
        #     "selected_items": selected_items,
        #     "total_value": sum(values[i] for i in selected_items),
        #     "total_weight": sum(weights[i] for i in selected_items)
        # }
        
        result = {"output": f"Model Result: {model.Items.pprint()}"}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import os
import json
from pyomo.environ import *
from pyomo.opt import SolverFactory
from utilities import read_data, create_instance, create_map
from opt_gurobipy import create_model #, get_vars_sol, create_df_coord, get_obj_components, create_df_OF
import gurobipy as gp
from gurobipy import GRB


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

data = {
    'Dataset 1': {'x': [1, 2, 3, 4, 5], 'y': [10, 15, 7, 12, 17]},
    'Dataset 2': {'x': [1, 2, 3, 4, 5], 'y': [5, 10, 15, 10, 5]},
    'Dataset 3': {'x': [1, 2, 3, 4, 5], 'y': [3, 6, 9, 12, 15]}
}

json_path = "data/data.json"
url_coord = 'https://docs.google.com/uc?export=download&id=1VYEnH735Tdgqe9cS4ccYV0OUxMqQpsQh'
url_dist = 'https://docs.google.com/uc?export=download&id=1Apbc_r3CWyWSVmxqWqbpaYEacbyf1wvV'
url_demand = 'https://docs.google.com/uc?export=download&id=1w0PMK36H4Aq39SAaJ8eXRU2vzHMjlWGe'
parameters = read_data(json_path, url_coord, url_dist, url_demand)
# instance = create_instance(parameters)        
# model = create_model(instance)
# model.setParam('MIPGap', 0.05) # Set the MIP gap tolerance to 5% (0.05)
# model.optimize()
# get solution
# var_sol = get_vars_sol(model)
# df_sol = create_df_coord(var_sol, df_coord)
# results_obj = get_obj_components(model)
# df_obj = create_df_OF(results_obj)


# def create_plot(df):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], mode='lines+markers', name='User Data'))
#     fig.update_layout(title='Plotly Line Graph', xaxis_title='X Axis', yaxis_title='Y Axis')
#     return fig.to_json(fig)

# def create_plot_from_dataset(dataset_name):
#     df = pd.DataFrame(data[dataset_name])
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines+markers', name=dataset_name))
#     fig.update_layout(title='Plotly Line Graph', xaxis_title='X Axis', yaxis_title='Y Axis')
#     return fig.to_json(fig)



@app.route('/update_graph', methods=['POST'])
def update_graph():
    dataset_name = request.json.get('dataset')
    fig = create_map(parameters['df_coord'])
    graph_json = fig.to_json(fig)
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
        fig = create_map(parameters['df_coord'])
        graph_json = fig.to_json(fig)
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

        # Solve using HiGHS solver
        solver = SolverFactory("appsi_highs")
        result = solver.solve(model)

        # Extract results
        selected_items = [i for i in model.Items if model.x[i].value == 1]

        # Return solution
        output = {
            "status": str(result.solver.status),
            "selected_items": selected_items,
            "total_value": sum(values[i] for i in selected_items),
            "total_weight": sum(weights[i] for i in selected_items)
        }
        
        result = {"output": f"Model Result: {output}"}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/run_sample_model', methods=['POST'])
def run_sample_model():
    inputs = request.get_json()
    parameters['enr'] = inputs['container_value']
    parameters['dep'] = inputs['deposit']
    parameters['qc'] = inputs['clasification']
    parameters['ql'] = inputs['washing']
    parameters['qa'] = inputs['transportation']
    instance = create_instance(parameters, seed=7)        
    model = create_model(instance)
    model.setParam('MIPGap', 0.05) # Set the MIP gap tolerance to 5% (0.05)
    model.optimize()
    result = model.ObjVal #sample_model(inputs)
    return jsonify({'result': result})

@app.route('/')
def index():
    fig = create_map(parameters['df_coord'])
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', graph_json=graph_json)

if __name__ == '__main__':
    app.run(debug=True)

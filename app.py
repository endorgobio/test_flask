from flask import Flask, render_template, request, jsonify
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import os
import json
from pyomo.environ import *
from pyomo.opt import SolverFactory
from utilities import read_data, create_instance, create_map, create_df_coord
from opt_gurobipy import create_model, get_vars_sol#, get_vars_sol, create_df_coord, get_obj_components, create_df_OF
import gurobipy as gp
from gurobipy import GRB


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

var_sol = {}

json_path = "data/data.json"
url_coord = 'https://docs.google.com/uc?export=download&id=1VYEnH735Tdgqe9cS4ccYV0OUxMqQpsQh'
url_dist = 'https://docs.google.com/uc?export=download&id=1Apbc_r3CWyWSVmxqWqbpaYEacbyf1wvV'
url_demand = 'https://docs.google.com/uc?export=download&id=1w0PMK36H4Aq39SAaJ8eXRU2vzHMjlWGe'
parameters = read_data(json_path, url_coord, url_dist, url_demand)
controls_default = {
    'container_value': 1240,
    'container_min': 1,
    'container_max': 10000,
    'deposit_value': 70,
    'deposit_min': 0,
    'deposit_max': 1250,
    'clasification_value': 140,
    'clasification_min': 0,
    'clasification_max': 1250,
    'washing_value': 210,
    'washing_min': 0,
    'washing_max': 1250,
    'transportation_value': 1,
    'transportation_min': 0,
    'transportation_max': 10,
    'transportation_step': 0.1
}
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
    

@app.route('/run_sample_model', methods=['POST'])
def run_sample_model():
    global var_sol

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
    var_sol = get_vars_sol(model)
    # df_coord = create_df_coord(var_sol, parameters['df_coord'])
    # # Convert dataframe to JSON
    # df_coord_json = df_coord.to_json(orient='records')
    # fig = create_map(df_coord)
    # graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # result = model.ObjVal #sample_model(inputs)
    # # return jsonify({'result': result})

    # # Return data and layout separately
    # return graph_json

    return jsonify({'result': True})

@app.route('/update_graph', methods=['POST'])
def update_graph():
    global var_sol

    df_coord = create_df_coord(var_sol, parameters['df_coord'])
    fig = create_map(df_coord)
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

@app.route('/')
def index():
    fig = create_map(parameters['df_coord'])
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', graph_json=graph_json, controls_default=controls_default)

if __name__ == '__main__':
    app.run(debug=True)

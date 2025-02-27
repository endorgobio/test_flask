from flask import Flask, render_template, request, jsonify
import plotly
import os
import json
from pyomo.environ import *
from pyomo.opt import SolverFactory
from utilities import read_data, create_instance, create_map, create_df_coord
from opt_pyomo import create_model_pyomo, get_vars_sol_pyomo#
from opt_gurobipy import create_model_gb, get_vars_sol_gb#, get_vars_sol, create_df_coord, get_obj_components, create_df_OF
import gurobipy as gp
from gurobipy import GRB



app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

opt_solution = {}
controls_default = {}

json_path = "data/data.json"
url_coord = 'https://docs.google.com/uc?export=download&id=1VYEnH735Tdgqe9cS4ccYV0OUxMqQpsQh'
url_dist = 'https://docs.google.com/uc?export=download&id=1Apbc_r3CWyWSVmxqWqbpaYEacbyf1wvV'
url_demand = 'https://docs.google.com/uc?export=download&id=1w0PMK36H4Aq39SAaJ8eXRU2vzHMjlWGe'
parameters = read_data(json_path, url_coord, url_dist, url_demand)


def get_controls_default(parameters):
    controls_default = {
        'container_value': parameters['enr'],
        'container_min': 0,
        'container_max': 10*parameters['enr'],
        'deposit_value': parameters['dep'],
        'deposit_min': 0,
        'deposit_max': 10*parameters['dep'],
        'clasification_value': parameters['qc'],
        'clasification_min': 0,
        'clasification_max': 10*parameters['qc'],
        'washing_value': parameters['ql'],
        'washing_min': 0,
        'washing_max': 10*parameters['ql'],
        'transportation_value': parameters['qa'],
        'transportation_min': 0,
        'transportation_max': 10*parameters['qa'],
        'transportation_step': 0.1
    }
    return controls_default


@app.route('/upload', methods=['POST'])
def upload_file():
    global parameters

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # TODO decide how to load coord and distances
        parameters = read_data(filepath, url_coord, url_dist, url_demand)
        fig = create_map(parameters['df_coord'])
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        controls_default = get_controls_default(parameters)
        return jsonify({'graph_json': graph_json, 'controls_default': controls_default})
        
    

@app.route('/run_sample_model', methods=['POST'])
def run_sample_model():

    inputs = request.get_json()
    parameters['enr'] = inputs['container_value']
    parameters['dep'] = inputs['deposit']
    parameters['qc'] = inputs['clasification']
    parameters['ql'] = inputs['washing']
    parameters['qa'] = inputs['transportation']
    instance = create_instance(parameters, seed=7)        
    # model = create_model_gb(instance)
    # model.setParam('MIPGap', 0.05) # Set the MIP gap tolerance to 5% (0.05)
    # model.optimize()
    # opt_solution['variables'] = get_vars_sol_gb(model)

    # solver = SolverFactory('gurobi')
    # solver.options['MIPGap'] = 0.01 
    solver = SolverFactory('appsi_highs')
    solver.options['mip_rel_gap'] = 0.01 
    model = create_model_pyomo(instance)
    solver.solve(model, tee=True)
    opt_solution['variables'] = get_vars_sol_pyomo(model)

    return jsonify({'result': True})

@app.route('/update_graph', methods=['POST'])
def update_graph():
    global opt_solution

    df_coord = create_df_coord(opt_solution['variables'], parameters['df_coord'])
    fig = create_map(df_coord)
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json



@app.route('/')
def index():
    controls_default = get_controls_default(parameters)
    fig = create_map(parameters['df_coord'])
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', graph_json=graph_json, controls_default=controls_default)

if __name__ == '__main__':
    app.run(debug=True)

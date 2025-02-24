
import pandas as pd
import requests
import json 
import pandas as pd
import re
import requests
import json
import numpy as np
# import gurobipy as gp
import plotly.graph_objects as go


        
def read_data(json_path, 
              url_coord, 
              url_dist, 
              url_demand):

    try:
        response = requests.get(json_path)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = json.loads(response.text)
        data['df_coord'] = pd.read_csv(url_coord) 
        data['df_dist'] = pd.read_csv(url_dist) 
        data['df_demand'] = pd.read_csv(url_demand) 
    except requests.exceptions.RequestException as e:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                data['df_coord'] = pd.read_csv(url_coord) 
                data['df_dist'] = pd.read_csv(url_dist) 
                data['df_demand'] = pd.read_csv(url_demand) 
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON content in file: {json_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the file: {e}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON content from URL: {json_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while fetching data from URL: {e}")
    
    return data




def select_nodes(df_coord, node_type, n):
    """
    Select and return a list of node IDs from a DataFrame based on type and sample size.
    The IDs are sorted such that numeric parts are sorted naturally (e.g., "A5" comes
    before "A50" and "A51").

    Parameters
    ----------
    df_coord : pd.DataFrame
        DataFrame containing node information, including 'type' and 'id' columns.
    node_type : str or int
        The type of node to filter by. Only nodes with this type will be selected.
    n : int
        The number of nodes to randomly sample from the filtered DataFrame.

    Returns
    -------
    list
        A sorted list of node IDs from the sample, with natural sorting applied.

    Examples
    --------
    >>> data = {'id': ['A5', 'A50', 'A51', 'A6'], 'type': ['A', 'A', 'A', 'A']}
    >>> df_coord = pd.DataFrame(data)
    >>> select_nodes(df_coord, 'A', 3)
    ['A5', 'A6', 'A50']
    """

    df = df_coord[df_coord['type'] == node_type]  # Filter by node type
    df = df.sample(n)                             # Sample 'n' nodes
    nodes = df['id'].tolist()                     # Convert 'id' column to list

    def custom_sort_key(s):
        """
        Custom sort key that extracts the numeric part of the string for natural sorting.

        Parameters
        ----------
        s : str
            The string to be sorted, containing both letters and numbers.

        Returns
        -------
        int
            The numeric part of the string for sorting.
        """
        return int(re.search(r'\d+', s).group())  # Extract and return numeric part of the string

    # Sort nodes based on numeric part of 'id'
    sorted_nodes = sorted(nodes, key=custom_sort_key)

    return sorted_nodes




def generate_demand(producers, packages, periodos, n_packings, dem_interval, 
                    demand_increment, initial_demand=None):
    """
    Generates initial demands and future demands for a set of producers, 
    packages, and periods with a demand increment.

    Args:
        producers (list): List of producers.
        packages (list): List of available package types.
        periodos (list): List of periods for which demands will be generated.
        n_packings (int): Number of packages selected by each producer.
        dem_interval (tuple): Demand range (minimum, maximum) for random generation.
        demand_increment (float): Annual demand increment
        initial_demand (dict, optional): Dictionary of initial demands, if provided. 
            If not provided, it will be generated randomly. Default is None.

    Returns:
        tuple: 
            - initial_demand (dict): Dictionary of generated or provided initial demands.
            - demands (dict): Dictionary of demands by package, producer, and period.
    """
    # define monthtly increment
    month_increment = (1 + demand_increment)**(1/12)-1
    # Create initial demands if none are provided
    if initial_demand is None:
        initial_demand = {}
        for producer in producers:
            chosen_packages = np.random.choice(packages, size=n_packings, replace=False)
            list_packages = []
            for p in chosen_packages:
                list_packages.append({p: np.random.randint(dem_interval[0], dem_interval[1])})
            initial_demand[producer] = list_packages

    demands = {}
    # Create demands for future periods according to the demand increment
    for producer, list_packages in initial_demand.items():
        for dict_pack in list_packages:
            for t in periodos:
                pack_id = list(dict_pack.keys())[0]
                demands[(pack_id, producer, t)] = int(dict_pack[pack_id] * (1 + month_increment) ** (t - 1))

    return initial_demand, demands

def read_dem_initial(df_demand):
    """
    Reads initial demand data from a DataFrame and organizes it by producer and packaging.

    Args:
        df_demand (pd.DataFrame): DataFrame containing demand data with columns for 
                                  'producer', 'packing', and 'demand'.

    Returns:
        dict: A dictionary where each producer is mapped to a list of dictionaries, 
              each containing a package and its corresponding demand.
    """
    demand_initial = {}

    for index, row in df_demand.iterrows():
        producer = row['producer']
        packing = row['packing']
        demand = row['demand']
        
        # Append the demand to the list for the producer
        if producer in demand_initial:
            demand_initial[producer].append({packing: demand})
        else:
            # Initialize the list for the producer if it doesn't exist
            demand_initial[producer] = [{packing: demand}]

    return demand_initial


def calculate_initialgeneration(initial_demand, packages):
    """
    Calculates the total initial generation of demand for each package type 
    based on the initial demand data.

    Args:
        initial_demand (dict): Dictionary where each producer is mapped to a list of 
                               dictionaries, each containing a package and its corresponding demand.
        packages (list): List of all possible package types.

    Returns:
        dict: A dictionary with each package as the key and the total initial demand 
              as the value. Packages with no demand are initialized to 0.
    """
    initial_generation = {}

    # Sum the demand for each package across all producers
    for producer, demands in initial_demand.items():
        for pack_demand in demands:
            for pack, demand in pack_demand.items():
                if pack in initial_generation:
                    initial_generation[pack] += demand
                else:
                    initial_generation[pack] = demand

    # Ensure all packages are included, even those with no demand
    for package in packages:
        if package not in initial_generation:
            initial_generation[package] = 0

    return initial_generation



def distribute_demand(n, total_demand):
    """
    Distributes a total demand into 'n' random parts, ensuring that the sum equals the total demand.

    Args:
        n (int): The number of parts to divide the demand into.
        total_demand (float): The total amount of demand to be distributed.

    Returns:
        np.ndarray: An array of 'n' random values that sum to the total demand.
    """
    # Generate n-1 random values in the interval (0, total_demand)
    cuts = np.sort(np.random.uniform(0, total_demand, n-1))

    # Add the boundaries 0 and total_demand to the cuts
    cuts = np.concatenate(([0], cuts, [total_demand]))

    # The differences between consecutive cuts give the random numbers
    random_numbers = np.diff(cuts)

    return random_numbers


def create_instance(parameters, seed=None):
    """Creates an instance for the optimization model using provided parameters.

    Args:
        parameters (dict): A dictionary containing various parameters for the instance, including:
            - df_coord: DataFrame containing node coordinates.
            - df_dist: DataFrame containing distances between nodes.
            - df_demand: DataFrame containing demand data.
            - n_acopios: Number of collection nodes.
            - n_centros: Number of classification centers.
            - n_plantas: Number of washing plants.
            - n_productores: Number of producers.
            - n_envases: Number of types of containers.
            - n_periodos: Number of periods for the model.
            - ccv: Classification capacity for each center.
            - acv: Storage capacity for each center.
            - lpl: Washing capacity for each plant.
            - apl: Storage capacity for each plant.
            - arr_cv: Rental cost for classification centers.
            - inflation: Anuual estimated inflation
            - ade_cv: Adaptation cost for classification centers.
            - ade_pl: Adaptation cost for washing plants.
            - dep: Deposit cost for containers.
            - enr: Price for returnable containers.
            - tri: Price for crushed containers.
            - adem: Annual demand increment rate.
            - initial_demand: Initial demand values.
            - recup: Initial recovery rate.
            - recup_increm: Annual Increment rate for recovery.
            - n_pack_prod: Number of packs produced.
            - dem_interval: Demand interval.
        seed: Random seed

    Returns:
        dict: A dictionary representing the optimization model instance with the following keys:
            - acopios: List of selected collection nodes.
            - centros: List of selected classification centers.
            - plantas: List of selected washing plants.
            - productores: List of selected producers.
            - envases: List of container types.
            - periodos: List of time periods.
            - cc: Classification capacities for centers.
            - ca: Storage capacities for centers.
            - cl: Washing capacities for plants.
            - cp: Storage capacities for plants.
            - rv: Rental costs for classification centers per period.
            - da: Distances from collection points to classification centers.
            - dl: Distances from classification centers to washing plants.
            - dp: Distances from washing plants to producers.
            - rl: Rental costs for washing plants per period.
            - av: Adaptation costs for classification centers per period.
            - al: Adaptation costs for washing plants per period.
            - qd: Deposit costs for containers.
            - pv: Prices for returnable containers.
            - pt: Prices for crushed containers.
            - b: Demand increment rates per period.
            - dem: Aggregate initial demand.
            - de: Periodic demand.
            - gi: Initial generation values.
            - ge: Generation per period.
            - iv: Initial inventory for classification centers.
            - il: Initial inventory for washing plants.
            - ci: Inventory costs for classification centers.
            - cv: Inventory costs for washing plants.
            - pe: Prices for new containers.
            - inflation: Annual inflation
    """
    
    # set random seed
    if seed != None:
        np.random.seed(seed)
    
    # Copy parameters into data for the instance
    instance = {k: v for k, v in parameters.items()}

    # Read dataframes
    df_coord = instance['df_coord']
    df_dist = instance['df_dist']
    df_demand = instance['df_demand']

    # Define sets
    acopios = select_nodes(df_coord, 'collection', instance['n_acopios'])
    centros = select_nodes(df_coord, 'clasification', instance['n_centros'])
    plantas = select_nodes(df_coord, 'washing', instance['n_plantas'])
    productores = select_nodes(df_coord, 'producer', instance['n_productores'])
    envases = ['E' + str(i) for i in range(instance['n_envases'])]
    periodos = [i + 1 for i in range(instance['n_periodos'])]

    instance['acopios'] = acopios
    instance['centros'] = centros
    instance['plantas'] = plantas
    instance['productores'] = productores
    instance['envases'] = envases
    instance['periodos'] = periodos

    instance['cc'] = {centro: instance['ccv'] for centro in centros}  # Classification capacity for centers
    instance['ca'] = {centro: instance['acv'] for centro in centros}  # Storage capacity for centers
    instance['cl'] = {planta: instance['lpl'] for planta in plantas}  # Washing capacity for plants
    instance['cp'] = {planta: instance['apl'] for planta in plantas}  # Storage capacity for plants
    

    # Rental cost for classification centers
    rv = {(c, 1): instance['arr_cv'] for c in centros}  
    for t in range(2, instance['n_periodos'] + 1):
       for c in centros:
           rv[(c, t)] = int(instance['arr_cv'] * (1 + instance['inflation'])**((t-1)//12)) # The price changes every year
    instance['rv'] = rv
    
    # Rental cost for washing plants
    rl = {(l, 1): instance['arr_pl'] for l in plantas}  
    for t in range(2, instance['n_periodos'] + 1):
        for l in plantas:
            rl[(l, t)] = int(instance['arr_pl'] * (1 + instance['inflation'])**((t-1)//12))
    instance['rl'] = rl
    
    # Adaptation cost for classification centers
    av = {(c, 1): instance['ade_cv'] for c in centros}  
    for t in range(2, instance['n_periodos'] + 1):
        for c in centros:
            av[(c, t)] = int(instance['ade_cv'] * (1 + instance['inflation'])**((t-1)//12))
    instance['av'] = av
    
    # Adaptation cost for washing plants
    al = {(l, 1): instance['ade_pl'] for l in plantas}  
    for t in range(2, instance['n_periodos'] + 1):
        for l in plantas:
            al[(l, t)] = int(instance['ade_pl'] * (1 + instance['inflation'])**((t-1)//12))
    instance['al'] = al
    
    # Distance from collection points to classification centers
    instance['da'] = {(a, c): df_dist[(df_dist['origin'] == a) & (df_dist['destination'] == c)][
        instance['type_distance']].item() for a in acopios for c in centros}
    
    # Distance from classification centers to washing plants
    instance['dl'] = {(c, l): df_dist[(df_dist['origin'] == c) & (df_dist['destination'] == l)][
        instance['type_distance']].item() for c in centros for l in plantas}
    
    # Distance from washing plants to producers
    instance['dp'] = {(l, p): df_dist[(df_dist['origin'] == l) & (df_dist['destination'] == p)][
        instance['type_distance']].item() for l in plantas for p in productores}

 



    instance['qd'] = {envase: instance['dep'] for envase in envases}  # Deposit cost
    instance['pv'] = {envase: instance['enr'] for envase in envases}  # Price for returnable containers
    instance['pt'] = {envase: instance['tri'] for envase in envases}  # Price for crushed containers
    instance['b'] = {t: instance['adem'] for t in range(1, instance['n_periodos'] + 1)}  # Demand increment rate

    # Generate demand and initial generation
    de_agg, de = generate_demand(productores, envases, periodos, instance['n_pack_prod'],
                                  instance['dem_interval'], instance['adem'], instance['initial_demand'])
    instance['dem'] = de_agg  # Initial demand
    instance['de'] = de  # Periodic demand
    gi = calculate_initialgeneration(de_agg, envases)  # Initial generation
    instance['gi'] = gi

    ge_agg = {}  # Aggregate generation
    for p in envases:
        for t in range(1, instance['n_periodos']):
            suma = 0
            for k in de.keys():
                if (k[0] == p and k[2] == t):
                    suma += de[k]
            ge_agg[(p, t + 1)] = suma
    
    month_incr_recup = (1+instance['recup_increm'])**(1/12) - 1
    a = {1: instance['recup']}
    for t in range(2, instance['n_periodos'] + 1):
        a[t] = min(1, a[t - 1] * (1 + month_incr_recup))
    instance['a'] = a  # Recovery rate

    ge = {}
    for p in envases:
        for t in range(1, instance['n_periodos'] + 1):
            if t > 1:
                dist = distribute_demand(instance['n_acopios'], ge_agg[(p, t)])
                for i in range(instance['n_acopios']):
                    ge[(p, acopios[i], t)] = dist[i] * a[t]
            else:
                dist = distribute_demand(instance['n_acopios'], gi[p])
                for i in range(instance['n_acopios']):
                    ge[(p, acopios[i], 1)] = dist[i] * a[t]
    instance['ge'] = ge  # Generation per period

    # Initial inventory and inventory costs
    instance['iv'] = {(e, c): 0 for e in envases for c in centros}  # Initial inventory for classification centers
    instance['il'] = {(e, l): 0 for e in envases for l in plantas}  # Initial inventory for washing plants
    instance['ci'] = {centro: instance['cinv'] for centro in centros}  # Inventory cost for centers
    instance['cv'] = {planta: instance['pinv'] for planta in plantas}  # Inventory cost for plants
    instance['pe'] = {envase: instance['envn'] for envase in envases}  # Price for new containers
    
    return instance



def distancia_geo(punto1: tuple, punto2: tuple) -> float:
    """
    Calcular la distancia de conducción entre dos puntos usando la API de OSRM.

    Args:
        punto1 (tuple): Coordenadas del primer punto (latitud, longitud).
        punto2 (tuple): Coordenadas del segundo punto (latitud, longitud).

    Returns:
        float: Distancia en kilómetros entre los dos puntos, o None si hay un error.
    """
    url = 'http://router.project-osrm.org/route/v1/driving/'
    o1 = f"{punto1[1]},{punto1[0]}"  # Invertir a (longitud, latitud) para OSRM
    o2 = f"{punto2[1]},{punto2[0]}"
    ruta = f"{o1};{o2}"
    
    response = requests.get(url + ruta)

    if response.status_code == 200:
        data = json.loads(response.content)
        return data['routes'][0]['legs'][0]['distance'] / 1000  # Convertir a km
    else:
        return None
    
    
    
def create_df_coord(var_sol, df_coord):
    active_act = []
    # active collection 
    df_q = var_sol['q']
    df_q = df_q[df_q['cantidad'] > 0.01]
    active_act.extend(list(df_q['acopio'].unique()))
    # active collection 
    df_y = var_sol['y']
    df_y = df_y[df_y['apertura'] > 0.01]
    active_act.extend(list(df_y['centro'].unique()))
    # active washing 
    df_w = var_sol['w']
    df_w = df_w[df_w['apertura'] > 0.01]
    active_act.extend(list(df_w['planta'].unique()))
    # active producer 
    df_u = var_sol['u']
    df_u = df_u[df_u['cantidad'] > 0.01]
    active_act.extend(list(df_u['productor'].unique()))
        
    df_sol =  df_coord[df_coord["id"].isin(active_act)]
    df_sol.reset_index(inplace=True)
    
    return df_sol



def create_df_OF(results_obj):
    df_obj = pd.DataFrame(list(results_obj.items()), columns=["Category", "Value"])
    # Function to determine the type based on the Category column
    def type_OF(row):
        if 'egreso' in row["Category"]:
            return "egreso"
        elif 'ingreso' in row["Category"]:
            return "ingreso"
        else:
            return "other"  # Default case if neither 'egreso' nor 'ingreso' is found

    # Apply the function row-wise to create a new column 'type'
    df_obj['Type'] = df_obj.apply(type_OF, axis=1)

    # Group by 'Type' and calculate the sum of 'Value' for each group
    grouped_df = df_obj.groupby('Type')['Value'].sum().reset_index()

    # Separate the total sums for 'egreso' and 'ingreso'
    total_egreso = grouped_df[grouped_df['Type'] == 'egreso']['Value'].iloc[0]
    total_ingreso = grouped_df[grouped_df['Type'] == 'ingreso']['Value'].iloc[0]

    # Create a new column to store the divided values
    df_obj['Percentage'] = df_obj.apply(
        lambda row: np.round(100*row['Value'] / total_egreso, 1) if row['Type'] == 'egreso' else 
                   (np.round(100*row['Value'] / total_ingreso, 1) if row['Type'] == 'ingreso' else None), axis=1)
    
    df_obj = df_obj.sort_values(by=['Type', 'Percentage'], ascending=[False, False])
    df_obj['%'] = df_obj['Percentage'].astype(str) + '%'
    df_obj['Name'] = df_obj['Category'].apply(lambda x: x.split('_')[-1] if len(x.split('_')) == 3 else x)
    df_obj = df_obj.reset_index(drop=True)
    
    return df_obj


def create_df_util(var_sol, parameters):
    # Use of the classification inventory capacity
    df_util = var_sol['ic'].groupby(['periodo']).agg(inv_class=("cantidad", "sum")).reset_index()
    df_util['inv_class'] = np.round(100*df_util['inv_class'] / (var_sol['y']['apertura'].sum()*parameters['acv']), 1)
    # Use of the washing inventory capacity
    df_temp = var_sol['ip'].groupby(['periodo']).agg(inv_wash=("cantidad", "sum")).reset_index()
    df_temp['inv_wash'] = np.round(100*df_temp['inv_wash'] / (var_sol['w']['apertura'].sum()*parameters['apl']), 1)
    df_util = pd.merge(df_util, df_temp, on="periodo", how="inner")
    # Use of the classification processing capacity
    df_temp = var_sol['r'].groupby(['periodo']).agg(cap_class=("cantidad", "sum")).reset_index()
    df_temp['cap_class'] = np.round(100*df_temp['cap_class'] / (var_sol['y']['apertura'].sum()*parameters['ccv']), 1)
    df_util = pd.merge(df_util, df_temp, on="periodo", how="inner")
    # Use of the classification processing capacity
    df_temp = var_sol['u'].groupby(['periodo']).agg(cap_wash=("cantidad", "sum")).reset_index()
    df_temp['cap_wash'] = np.round(100*df_temp['cap_wash'] / (var_sol['w']['apertura'].sum()*parameters['lpl']), 1)
    df_util = pd.merge(df_util, df_temp, on="periodo", how="inner")
    df_util["periodo"] = df_util["periodo"].astype(int)
    df_util = df_util.sort_values(by="periodo", ascending=True).reset_index(drop=True)
    
    return df_util
    
def create_map(df):
    """
    Creates a scatter mapbox visualization based on the data provided in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing at least the following columns:
        - 'type': Categorical values used to group markers.
        - 'latitude': Latitude coordinates for each marker.
        - 'longitude': Longitude coordinates for each marker.
        - 'id': Identifier text for each marker.

    Returns:
    --------
    plotly.graph_objects.Figure
        A Plotly figure object representing the scatter mapbox.
    """
    # Initialize the map figure
    map_actors = go.Figure()

    # Extract unique actor types from the 'type' column
    actors = list(df['type'].unique())

    # Handle case where no actors are present
    if len(actors) == 0:
        map_actors.add_trace(go.Scattermapbox(
            lat=[],
            lon=[],
            mode='markers',
        ))

    # Loop through each actor type and add corresponding markers
    for actor in actors:
        df_filter = df[df['type'] == actor]

        # Add markers for the current actor
        map_actors.add_trace(go.Scattermapbox(
            lat=df_filter['latitude'],
            lon=df_filter['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=12,
                opacity=0.8
            ),
            text=df_filter['id'],
            textposition='top right',
            name=actor
        ))

    # Configure the layout of the map
    map_actors.update_layout(
        mapbox=dict(
            style="open-street-map",  # Use OpenStreetMap style
            center=dict(lat=6.261943611002649, lon=-75.58979925245441),  # Center on specific coordinates
            zoom=4  # Set default zoom level
        ),
        autosize=True,
        hovermode='closest',
        showlegend=True,
        height=600,  # Set map height in pixels
        width=800
    )

    return map_actors

def graph_costs(df):
    # Create the bar chart
    fig = go.Figure()

    # Add horizontal bars costs
    df['Value'] = np.round(df['Value']/1000000, 1)
    df_filter = df[df['Type']=='egreso']
    fig.add_trace(go.Bar(
        x=df_filter['Value'],
        y=df_filter['Name'],
        orientation='h',  # Horizontal orientation
        marker_color='#73C2FB',  # Bar color
        opacity=0.9,  # Bar opacity
        name='costs',
        text=df_filter['%'],  # Dynamically formatted text
        textposition='outside',  # Position of labels
        showlegend=False
    ))

    fig.add_trace(go.Bar(
        x=[0], y=[" "],
        orientation='h',  # Horizontal orientation
        marker_color="white",
        name='Space',
        showlegend=False
    ))


    # Add horizontal bars income
    df_filter = df[df['Type']=='ingreso']
    fig.add_trace(go.Bar(
        x=df_filter['Value'],
        y=df_filter['Name'],
        orientation='h',  # Horizontal orientation
        marker_color='#002D62',  # Bar color
        opacity=0.9,  # Bar opacity
        name='income',
        text=df_filter['%'],  # Dynamically formatted text
        textposition='outside',  # Position of labels
        showlegend=False
    ))



    # Add a vertical line at 100%
    # fig.add_shape(
    #     type="line",
    #     x0=100, x1=100,
    #     y0=-0.5, y1=len(categories) + len(categories2) - 0.5,  # Extend the line across the bar chart
    #     line=dict(color="grey", width=2),
    #     name='Threshold'
    # )

    # Add custom legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color="#002D62"),
        name="Ingresos"
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color="#73C2FB"),
        name="Egresos"
    ))

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Valor (millones)",
        yaxis_title=" ",
        #xaxis=dict(ticksuffix="%"),  # Add % suffix to x-axis
        template="plotly_white",
        showlegend=True
    )
    
    return fig
    
def graph_utilization(df):
     # Create the figure
     fig = go.Figure()

     # Add lines for each column
     fig.add_trace(go.Scatter(x=df["periodo"], y=df["inv_class"], mode='lines', name='inv. clasificación'))
     fig.add_trace(go.Scatter(x=df["periodo"], y=df["inv_wash"], mode='lines', name='inv. lavado'))
     fig.add_trace(go.Scatter(x=df["periodo"], y=df["cap_class"], mode='lines', name='prod. clasificación'))
     fig.add_trace(go.Scatter(x=df["periodo"], y=df["cap_wash"], mode='lines', name='prod. lavado'))

     # Update layout
     fig.update_layout(
         title=" ",
         xaxis_title="periodo",
         yaxis_title="% de uso",
         legend_title=" ",
         template="plotly_white",
     )
     return fig 

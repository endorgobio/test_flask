import pandas as pd
from pyomo.environ import *



def create_model_pyomo(instance,
                 model_integer = False # when True the model considers integer variables (False by default)
                 ):

# read instance data
    n_acopios = instance['n_acopios']
    n_centros = instance['n_centros']
    n_plantas = instance['n_plantas']
    n_productores = instance['n_productores']
    n_envases = instance['n_envases']
    n_periodos = instance['n_periodos']
    inflation = instance['inflation']
    qc = instance['qc']
    qt = instance['qt']
    ql = instance['ql']
    qb = instance['qb']
    ta = instance['ta']
    tl = instance['tl']
    qa = instance['qa']
    wa = instance['wa']
    em = instance['em']
    el = instance['el']
    et = instance['et']
    en = instance['en']
    recup_increm = instance['recup_increm']
    ccv = instance['ccv']
    acv = instance['acv']
    lpl = instance['lpl']
    apl = instance['apl']
    # dimdist = instance['dimdist']
    arr_cv = instance['arr_cv']
    arr_pl = instance['arr_pl']
    ade_cv = instance['ade_cv']
    ade_pl = instance['ade_pl']
    dep = instance['dep']
    enr = instance['enr']
    tri = instance['tri']
    adem = instance['adem']
    # demanda = instance['demanda']
    recup = instance['recup']
    cinv = instance['cinv']
    pinv = instance['pinv']
    envn = instance['envn']
    
    
    # Assign sets and other generated variables from the dictionary
    acopios = instance['acopios']
    centros = instance['centros']
    plantas = instance['plantas']
    productores = instance['productores']
    envases = instance['envases']
    periodos = instance['periodos']
    
    # Assign parameters and calculations from the dictionary
    cc = instance['cc']
    ca = instance['ca']
    cp = instance['cp']
    cl = instance['cl']
    # coord_acopios = instance['coord_acopios']
    # coord_centros = instance['coord_centros']
    # coord_plantas = instance['coord_plantas']
    # coord_productores = instance['coord_productores']
    da = instance['da']
    dl = instance['dl']
    dp = instance['dp']
    rv = instance['rv']
    rl = instance['rl']
    av = instance['av']
    al = instance['al']
    qd = instance['qd']
    pv = instance['pv']
    pt = instance['pt']
    b = instance['b']
    # env_pdtor = instance['env_pdtor']
    dem = instance['dem']
    de = instance['de']
    gi = instance['gi']
    ge = instance['ge']
    iv = instance['iv']
    il = instance['il']
    ci = instance['ci']
    cv = instance['cv']
    pe = instance['pe']
    a = instance['a']
    
    
    model = ConcreteModel('CircularEconomy')
        
    # Definir los conjuntos
    model.envases = Set(initialize=envases)
    model.acopios = Set(initialize=acopios)
    model.centros = Set(initialize=centros)
    model.plantas = Set(initialize=plantas)
    model.periodos = Set(initialize=periodos)
    model.productores = Set(initialize=productores)

    #Parametros
    
    # model.de = Set(initialize=de.keys())
    model.de = Param(model.envases, model.productores, model.periodos, initialize=de, default=0, within=NonNegativeReals)
    model.ta = Param(within=NonNegativeReals, initialize=ta)
    model.tl = Param(within=NonNegativeReals, initialize=tl)
    model.iv = Param(model.envases,model.centros,initialize=iv)
    model.cc = Param(model.centros,initialize=cc)
    model.ca = Param(model.centros,initialize=ca)
    model.cl = Param(model.plantas,initialize=cl)
    model.il = Param(model.envases,model.plantas,initialize=il)
    model.cp = Param(model.plantas,initialize=cp)
    model.ge = Param(model.envases,model.acopios,model.periodos,initialize=ge)
    model.pv = Param(model.envases,initialize=pv)
    model.inflation = Param(within=NonNegativeReals, initialize=inflation)
    model.wa = Param(within=NonNegativeReals, initialize=wa)
    model.pt = Param(model.envases,initialize=pt)
    model.av = Param(model.centros,model.periodos,initialize=av)
    model.al = Param(model.plantas,model.periodos,initialize=al)
    model.rv = Param(model.centros,model.periodos,initialize=rv)
    model.rl = Param(model.plantas,model.periodos,initialize=rl)
    model.qa = Param(within=NonNegativeReals, initialize=qa)
    model.da = Param(model.acopios,model.centros,initialize=da)
    model.dl = Param(model.centros,model.plantas,initialize=dl)
    model.dp = Param(model.plantas,model.productores,initialize=dp)
    model.qd = Param(model.envases,initialize=qd)
    model.qc = Param(within=NonNegativeReals, initialize=qc)
    model.ql = Param(within=NonNegativeReals, initialize=ql)
    model.qb = Param(within=NonNegativeReals, initialize=qb)
    model.qt = Param(within=NonNegativeReals, initialize=qt)
    model.ci = Param(model.centros, initialize=ci)
    model.cv = Param(model.plantas, initialize=cv)
    model.em = Param(within=NonNegativeReals, initialize=em)
    model.el = Param(within=NonNegativeReals, initialize=el)
    model.et = Param(within=NonNegativeReals, initialize=et)
    
    # Definir las variables
    model.x = Var(centros, periodos, domain=Binary, name="x")
    model.y = Var(centros, periodos, domain=Binary, name="y")
    model.z = Var(plantas, periodos, domain=Binary, name="z")
    model.w = Var(plantas, periodos, domain=Binary, name="w")
    
    # Si model_integer es verdadero
    if model_integer:
        # Variables enteras
        model.q = Var(model.envases, model.acopios, model.centros, model.periodos, domain=Integers, name="q")
        model.r = Var(model.envases, model.centros, model.plantas, model.periodos, domain=Integers, name="r")
        model.combinations_u = Set(initialize=[(p, k, l, t) for p, l, t in model.de for k in model.plantas])
        model.u = Var(model.combinations_u, domain=Integers, name="u")
        model.ic = Var(model.envases, model.centros, model.periodos, domain=Integers, name="ic")
        model.ip = Var(model.envases, model.plantas, model.periodos, domain=Integers, name="ip")
        model.combinations_er = Set(initialize=[(p, l, t) for p, l, t in model.de])
        # model.er = Var(model.combinations_er, domain=Integers, name="er")
    else:
        # Variables continuas
        model.q = Var(model.envases, model.acopios, model.centros, model.periodos, domain=NonNegativeReals, name="q")
        model.r = Var(model.envases, model.centros, model.plantas, model.periodos, domain=NonNegativeReals, name="r")
        model.combinations_u = Set(initialize=[(p, k, l, t) for p, l, t in model.de for k in model.plantas])
        model.u = Var(model.combinations_u, domain=NonNegativeReals, name="u")
        model.ic = Var(model.envases, model.centros, model.periodos, domain=NonNegativeReals, name="ic")
        model.ip = Var(model.envases, model.plantas, model.periodos, domain=NonNegativeReals, name="ip")
        model.combinations_er = Set(initialize=[(p, l, t) for p, l, t in model.de])
        # model.er = Var(model.combinations_er, domain=NonNegativeReals, name="er")
    
    
    # **FUNCIÓN OBJETIVO**
    # Componente ingreso_retornable
    
    model._ingreso_retornable = sum(
    model.u[p, k, l, t] * model.pv[p] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t 
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    )
    
    # Componente ingreso_triturado
    model._ingreso_triturado = (
    sum(
    model.r[p, j, k, t] * (1 - model.ta) / model.ta * model.pt[p] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for j in model.centros 
    for k in model.plantas 
    for t in model.periodos
    )
    + sum(
    model.u[p, k, l, t] * (1 - model.tl) / model.tl * model.pt[p] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    )
    )
    
    # Componente egreso_adecuar
    model._egreso_adecuar = (
    sum(
    model.y[j, t] * model.av[j, t] / (1 + model.wa)**t 
    for j in model.centros 
    for t in model.periodos
    )
    + sum(
    model.w[k, t] * model.al[k, t] / (1 + model.wa)**t 
    for k in model.plantas 
    for t in model.periodos
    )
    )
    
    # Componente egreso_uso
    model._egreso_uso = (
    sum(
    model.x[j, t] * model.rv[j, t] / (1 + model.wa)**t 
    for j in model.centros 
    for t in model.periodos
    )
    + sum(
    model.z[k, t] * model.rl[k, t] / (1 + model.wa)**t 
    for k in model.plantas 
    for t in model.periodos
    )
    )
    
    # Componente egreso_transporte
    model._egreso_transporte = (
    sum(
    model.q[p, i, j, t] * model.qa * model.da[i, j] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for i in model.acopios 
    for j in model.centros 
    for t in model.periodos
    )
    + sum(
    model.r[p, j, k, t] * model.qa * model.dl[j, k] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for j in model.centros 
    for k in model.plantas 
    for t in model.periodos
    )
    + sum(
    model.u[p, k, l, t] * model.qa * model.dp[k, l] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    )
    )
    
    # Componente egreso_compra
    model._egreso_compra = sum(
    model.q[p, i, j, t] * model.qd[p] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for i in model.acopios 
    for j in model.centros 
    for t in model.periodos
    )
    
    # Componente egreso_inspeccion
    model._egreso_inspeccion = (
    sum(
    (model.r[p, j, k, t] / model.ta) * model.qc * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for j in model.centros 
    for k in model.plantas 
    for t in model.periodos
    )
    + sum(
    (model.u[p, k, l, t] / model.tl) * model.qc * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    )
    )
    
    # Componente egreso_lavado
    model._egreso_lavado = sum(
    (model.u[p, k, l, t] / model.tl) * model.ql * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    )
    
    # Componente egreso_pruebas
    model._egreso_pruebas = sum(
    model.u[p, k, l, t] * model.qb * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    )
    
    # Componente egreso_trituracion
    model._egreso_trituracion = (
    sum(
    model.r[p, j, k, t] * (1 - model.ta) / model.ta * model.qt * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for j in model.centros 
    for k in model.plantas 
    for t in model.periodos
    )
    + sum(
    model.u[p, k, l, t] * (1 - model.tl) / model.tl * model.qt * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    )
    )

    
    # Componente egreso_invcentros
    model._egreso_invcentros = sum(
    model.ic[p, j, t] * model.ci[j] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for j in model.centros 
    for t in model.periodos
    )
    
    # Componente egreso_invplantas
    model._egreso_invplantas = sum(
    model.ip[p, k, t] * model.cv[k] * (1 + model.inflation)**((t-1)//12) / (1 + model.wa)**t
    for p in model.envases 
    for k in model.plantas 
    for t in model.periodos
    )
    
    # Componente emisiones_transporte
    model._emisiones_transporte = (
    sum(
    model.da[i, j] * model.q[p, i, j, t]
    for p in model.envases 
    for i in model.acopios 
    for j in model.centros 
    for t in model.periodos
    )
    + sum(
    model.dl[j, k] * model.r[p, j, k, t]
    for p in model.envases 
    for j in model.centros 
    for k in model.plantas 
    for t in model.periodos
    )
    + sum(
    model.dp[k, l] * model.u[p, k, l, t]
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    ) * model.em
    )
    
    # Componente emisiones_lavado
    model._emisiones_lavado = sum(
    (model.u[p, k, l, t] / model.tl) * model.el
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos 
    if (p, k, l, t) in model.combinations_u
    )
    
    # Componente emisiones_trituracion
    model._emisiones_trituracion = (
    sum(
    model.r[p, j, k, t] * (1 - model.ta) / model.ta * model.et
    for p in model.envases 
    for j in model.centros 
    for k in model.plantas 
    for t in model.periodos
    )
    + sum(
    model.u[p, k, l, t] * (1 - model.tl) / model.tl * model.et
    for p in model.envases 
    for k in model.plantas 
    for l in model.productores 
    for t in model.periodos
    if (p, k, l, t) in model.combinations_u
    )
    )
    
    # Definir la función objetivo
    funcion_objetivo = (
    model._ingreso_retornable + model._ingreso_triturado - model._egreso_adecuar - model._egreso_uso -
    model._egreso_transporte - model._egreso_compra - model._egreso_inspeccion - model._egreso_lavado -
    model._egreso_pruebas - model._egreso_trituracion - model._egreso_invcentros - model._egreso_invplantas
    )
    
    # Establecer la función objetivo del modelo
    model.objVal = Objective(expr=funcion_objetivo, sense=maximize)
    

    
    # Restricción 1: Capacidad de procesamiento centro de clasificación
    def cap_proc_centros_rule(model, j, t):
        return sum(model.r[p, j, k, t] / model.ta for p in model.envases for k in model.plantas) <= model.cc[j] * model.x[j, t]
    model.cap_proc_centros = Constraint(model.centros, model.periodos, rule=cap_proc_centros_rule)
    
    # Restricción 2: Capacidad de procesamiento plantas de lavado
    def cap_proc_plantas_rule(model, k, t):
        return sum(model.u[p, k, l, t] / model.tl for p in model.envases for l in model.productores if (p, k, l, t) in model.combinations_u
        ) <= model.cl[k] * model.z[k, t]
    model.cap_proc_plantas = Constraint(model.plantas, model.periodos, rule=cap_proc_plantas_rule)
    
    # Restricción 3: Cumplimiento de demanda
    def cumplimiento_demanda_rule(model, p, l, t):
        if (p,l,t) in model.de: 
            return sum(model.u[p, k, l, t] for k in model.plantas if (p, k, l, t) in model.combinations_u)  <= model.de[p, l, t]
        else:
            return Constraint.Skip
    model.cumplimiento_demanda = Constraint(model.envases, model.productores, model.periodos, rule=cumplimiento_demanda_rule)
    
    
    # Restricción 4: No debe recogerse más de la generación
    def no_recoger_mas_gen_rule(model, p, i, t):
        return sum(model.q[p, i, j, t] for j in model.centros) <= ge[p, i, t]
    model.no_recoger_mas_gen = Constraint(model.envases, model.acopios, model.periodos, rule=no_recoger_mas_gen_rule)
    
    # Restricción 5: Adecuación y apertura de centros de clasificación
    def mantener_abierto_centro_rule(model, j, t, tp):
        return model.x[j, t] >= model.y[j, tp]
    model.mantener_abierto_centro = Constraint(model.centros, model.periodos, model.periodos, rule=mantener_abierto_centro_rule)
    
    # Restricción 6: Usar cuando el centro está abierto
    def usar_cuando_centro_rule(model, j, t):
        return sum(model.y[j, tp] for tp in range(1, t+1)) >= model.x[j, t]
    model.usar_cuando_centro = Constraint(model.centros, model.periodos, rule=usar_cuando_centro_rule)
    
    # Restricción 7: Adecuar el centro (solo puede abrirse una vez)
    def adecuar_centro_rule(model, j):
        return sum(model.y[j, t] for t in model.periodos) <= 1
    model.adecuar_centro = Constraint(model.centros, rule=adecuar_centro_rule)
    
    # Restricción 8: Mantener abierta la planta de lavado
    def mantener_abierta_planta_rule(model, k, t, tp):
        return model.z[k, t] >= model.w[k, tp]
    model.mantener_abierta_planta = Constraint(model.plantas, model.periodos, model.periodos, rule=mantener_abierta_planta_rule)
    
    # Restricción 9: Usar cuando la planta está abierta
    def usar_cuando_planta_rule(model, k, t):
        return sum(model.w[k, tp] for tp in range(1, t+1)) >= model.z[k, t]
    model.usar_cuando_planta = Constraint(model.plantas, model.periodos, rule=usar_cuando_planta_rule)
    
    # Restricción 10: Adecuar la planta (solo puede abrirse una vez)
    def adecuar_planta_rule(model, k):
        return sum(model.w[k, t] for t in model.periodos) <= 1
    model.adecuar_planta = Constraint(model.plantas, rule=adecuar_planta_rule)
    
    # Restricción 11: Inventario en centros de clasificación
    def inv_centros_rule(model, p, j, t):
        if t > 1:
            return model.ic[p, j, t] == model.ic[p, j, t-1] + sum(model.q[p, i, j, t] for i in model.acopios) - sum(model.r[p, j, k, t] / model.ta for k in model.plantas)
        else:
            return model.ic[p, j, t] == model.iv[p, j] + sum(model.q[p, i, j, t] for i in model.acopios) - sum(model.r[p, j, k, t] / model.ta for k in model.plantas)
    model.inv_centros = Constraint(model.envases, model.centros, model.periodos, rule=inv_centros_rule)
    
    # Restricción 12: Capacidad de almacenamiento en centros de clasificación
    def cap_alm_centros_rule(model, j, t):
        return sum(model.ic[p, j, t] for p in model.envases) <= model.ca[j] * model.x[j, t]
    model.cap_alm_centros = Constraint(model.centros, model.periodos, rule=cap_alm_centros_rule)
    
    # Restricción 13: Inventario en plantas de lavado
    def inv_plantas_rule(model, p, k, t):
        if t > 1:
            return model.ip[p, k, t] == model.ip[p, k, t-1] + sum(model.r[p, j, k, t] for j in model.centros) - sum(model.u[p, k, l, t] / model.tl for l in model.productores if (p, k, l, t) in model.combinations_u)
        else:
            return model.ip[p, k, t] == model.il[p, k] + sum(model.r[p, j, k, t] for j in model.centros) - sum(model.u[p, k, l, t] / model.tl for l in model.productores if (p, k, l, t) in model.combinations_u)
    model.inv_plantas = Constraint(model.envases, model.plantas, model.periodos, rule=inv_plantas_rule)
    
    # Restricción 14: Capacidad de almacenamiento en plantas de lavado
    def cap_alm_plantas_rule(model, k, t):
        return sum(model.ip[p, k, t] for p in model.envases) <= model.cp[k] * model.z[k, t]
    model.cap_alm_plantas = Constraint(model.plantas, model.periodos, rule=cap_alm_plantas_rule)
    
    
    return model


def get_vars_sol_pyomo(model):
    
   
    dict_df = {}

    # Iterate over all variables in the model
    for var in model.component_objects(Var, active=True):
        var_name = var.name  # Name of the variable (e.g., 'x', 'y', 'z')

        # Store variable values and their indexes
        var_values = []

        for index in var:
            var_values.append(list(index) + [value(var[index])])  # Extract index values + variable value

        # Organize into dictionary by variable name
        if var_values:
            dict_df[var_name] = var_values

    # Predefined column names for each variable group
    col_names = {
        'x': ['centro', 'periodo', 'uso'],
        'y': ['centro', 'periodo', 'apertura'],
        'z': ['planta', 'periodo', 'uso'],
        'w': ['planta', 'periodo', 'apertura'],
        'q': ['envase', 'acopio', 'centro', 'periodo', 'cantidad'],
        'r': ['envase', 'centro', 'planta', 'periodo', 'cantidad'],
        'u': ['envase', 'planta', 'productor', 'periodo', 'cantidad'],
        'ic': ['envase', 'centro', 'periodo', 'cantidad'],
        'ip': ['envase', 'planta', 'periodo', 'cantidad'],
        'er': ['envase', 'productor', 'periodo', 'cantidad']
    }

    # Convert lists into DataFrames with proper column names
    dict_df = {key: pd.DataFrame(value, columns=col_names.get(key, ["index_" + str(i) for i in range(len(value[0]))])) 
               for key, value in dict_df.items()}

    return dict_df

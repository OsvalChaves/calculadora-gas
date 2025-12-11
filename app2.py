import streamlit as st
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Cálculo y Verificación SEP-10/SEP-30", layout="wide", page_icon="⚡")

# --- 2. BASE DE DATOS Y CONSTANTES ---
COMPONENTS_DB = {
    "N2": {"Tc": 126.2, "Pc": 33.9, "w": 0.037, "M": 28.01},
    "CO2": {"Tc": 304.1, "Pc": 73.8, "w": 0.225, "M": 44.01},
    "H2S": {"Tc": 373.2, "Pc": 89.4, "w": 0.099, "M": 34.08},
    "CH4": {"Tc": 190.6, "Pc": 46.0, "w": 0.011, "M": 16.04},
    "C2H6": {"Tc": 305.4, "Pc": 48.8, "w": 0.099, "M": 30.07},
    "C3H8": {"Tc": 369.8, "Pc": 42.5, "w": 0.152, "M": 44.10},
    "iC4H10": {"Tc": 408.2, "Pc": 36.5, "w": 0.183, "M": 58.12},
    "nC4H10": {"Tc": 425.2, "Pc": 38.0, "w": 0.199, "M": 58.12},
    "iC5H12": {"Tc": 460.4, "Pc": 33.8, "w": 0.227, "M": 72.15},
    "nC5H12": {"Tc": 469.7, "Pc": 33.7, "w": 0.251, "M": 72.15},
    "C6H14": {"Tc": 507.5, "Pc": 30.1, "w": 0.299, "M": 86.18},
    "C7H16": {"Tc": 540.2, "Pc": 27.4, "w": 0.349, "M": 100.21},
    "C8H18": {"Tc": 568.8, "Pc": 24.9, "w": 0.398, "M": 114.23},
    "C9H20+": {"Tc": 594.6, "Pc": 22.9, "w": 0.445, "M": 128.26},
    "O2": {"Tc": 154.6, "Pc": 50.4, "w": 0.022, "M": 31.99},
    "H2O": {"Tc": 647.1, "Pc": 220.6, "w": 0.344, "M": 18.015}
}

# Diámetros internos (pulgadas) para cañerías
PIPES_DB = {
    "3 inch": {"Sch40": 3.068, "Sch80": 2.900},
    "4 inch": {"Sch40": 4.026, "Sch80": 3.826},
    "6 inch": {"Sch40": 6.065, "Sch80": 5.761},
    "8 inch": {"Sch40": 7.981, "Sch80": 7.625},
    "10 inch": {"Sch40": 10.02, "Sch80": 9.562},
    "12 inch": {"Sch40": 12.00, "Sch80": 11.374},
}

# Límites definidos por el cliente (Hard Constraints)
LIMITS = {
    "SEP-10": {"Gas_Max": 2500000.0, "Liq_Max": 1200.0},  # 48x15
    "SEP-30": {"Gas_Max": 200000.0, "Liq_Max": 700.0}    # 36x15
}

# Volúmenes fijos por weir plate (m3)
VOLUMES_FIXED = {
    "SEP-10": {"40%": 1.47, "50%": 1.967, "60%": 2.465},
    "SEP-30": {"40%": 0.827, "50%": 1.107, "60%": 1.386}
}

# --- 3. FUNCIONES DE CÁLCULO ---
def calc_peng_robinson(T_k, P_bar, composition_mole_frac):
    R = 8.314462618
    P_pa = P_bar * 1e5
    M_mix = 0
    params = {}
    
    for comp, y in composition_mole_frac.items():
        if y == 0: continue
        d = COMPONENTS_DB[comp]
        M_mix += y * d['M']
        Tr = T_k / d['Tc']
        k = 0.37464 + 1.54226*d['w'] - 0.26992*(d['w']**2)
        alpha = (1 + k * (1 - np.sqrt(Tr)))**2
        a_i = 0.45724 * (R**2 * d['Tc']**2) / (d['Pc']*1e5) * alpha
        b_i = 0.07780 * (R * d['Tc']) / (d['Pc']*1e5)
        params[comp] = {'y': y, 'a': a_i, 'b': b_i}

    a_mix, b_mix = 0, 0
    for c1, p1 in params.items():
        b_mix += p1['y'] * p1['b']
        for c2, p2 in params.items():
            a_mix += p1['y'] * p2['y'] * np.sqrt(p1['a'] * p2['a'])
            
    A = (a_mix * P_pa) / (R * T_k)**2
    B = (b_mix * P_pa) / (R * T_k)
    c2 = -(1 - B)
    c1 = (A - 3*B**2 - 2*B)
    c0 = -(A*B - B**2 - B**3)
    
    roots = np.roots([1, c2, c1, c0])
    real_roots = roots[np.isreal(roots)].real
    Z = max(real_roots) if len(real_roots) > 0 else 1.0
    rho_kg_m3 = (P_pa * (M_mix / 1000)) / (Z * R * T_k)
    return Z, rho_kg_m3, M_mix

def get_closest_flange(id_req_mm):
    # Retorna el DN standard (en pulgadas) que tenga un bore > id_req
    # Simplificado ANSI 600 aprox ID
    stds = [
        (3, 2.9), (4, 3.826), (6, 5.761), 
        (8, 7.625), (10, 9.562), (12, 11.37)
    ]
    id_req_inch = id_req_mm / 25.4
    for dn, bore in stds:
        if bore >= id_req_inch:
            return f"{dn} inch"
    return ">12 inch"

# --- 4. PRESETS ---
PRESET_PETROLEO = {"N2":1.94,"CO2":0.138,"CH4":65.5,"C2H6":16.64,"C3H8":10.0,"iC4H10":0.61,"nC4H10":2.61,"iC5H12":0.71,"nC5H12":0.96,"C6H14":0.57,"C7H16":0.31,"C8H18":0.03,"C9H20+":0.0,"O2":0.0,"H2O":0.0}
PRESET_GAS = {"N2":0.204,"CO2":0.748,"CH4":94.87,"C2H6":3.721,"C3H8":0.376,"iC4H10":0.039,"nC4H10":0.028,"iC5H12":0.0,"nC5H12":0.0,"C6H14":0.015,"C7H16":0.0,"C8H18":0.0,"C9H20+":0.0,"O2":0.0,"H2O":0.0}

if 'df_comp' not in st.session_state:
    data = {"Componente": list(COMPONENTS_DB.keys()), "% Molar": [0.0]*len(COMPONENTS_DB)}
    st.session_state.df_comp = pd.DataFrame(data)
    st.session_state.df_comp.loc[st.session_state.df_comp['Componente']=='CH4', '% Molar'] = 100.0

if 'liq_density_input' not in st.session_state:
    st.session_state['liq_density_input'] = 850.0

# --- 5. LOGICA DE INTERFAZ ---

def load_petroleo():
    for i, row in st.session_state.df_comp.iterrows():
        st.session_state.df_comp.at[i, '% Molar'] = PRESET_PETROLEO.get(row['Componente'], 0.0)
    st.session_state['liq_density_input'] = 850.0 # Valor nominal Oil

def load_gas():
    for i, row in st.session_state.df_comp.iterrows():
        st.session_state.df_comp.at[i, '% Molar'] = PRESET_GAS.get(row['Componente'], 0.0)
    # Por punto 4: Asumir densidad de 830 kg/m3 para pozos de gas (agua)
    st.session_state['liq_density_input'] = 830.0 

st.title("🏭 Verificación de Equipos: SEP-10 & SEP-30")

# --- SELECCION DE EQUIPO Y POZO ---
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    equip_sel = st.selectbox("Equipo", ["SEP-10 (48x15)", "SEP-30 (36x15)"])
with col_sel2:
    well_type = st.radio("Tipo de Pozo", ["Pozo de Gas", "Pozo de Petróleo"], horizontal=True)

# Lógica de Constantes según selección
if "SEP-10" in equip_sel:
    EQ_KEY = "SEP-10"
    D_inch, L_ft = 48.0, 15.0
else:
    EQ_KEY = "SEP-30"
    D_inch, L_ft = 36.0, 15.0

# --- COMPOSICION ---
st.sidebar.header("Composición")
col_b1, col_b2 = st.sidebar.columns(2)
col_b1.button("Cargar Petróleo", on_click=load_petroleo)
col_b2.button("Cargar Gas", on_click=load_gas)
edited_df = st.sidebar.data_editor(st.session_state.df_comp, hide_index=True, height=350)
st.session_state.df_comp = edited_df
z_comp = {r['Componente']: r['% Molar']/100 for i, r in edited_df.iterrows()}

# --- CONDICIONES DE PROCESO ---
st.header("Condiciones Operativas")
c1, c2, c3, c4 = st.columns(4)
with c1:
    t_val = st.number_input("Temperatura (°C)", value=45.0)
with c2:
    p_val = st.number_input("Presión (bar g)", value=90.0)
with c3:
    res_time = st.number_input("Tiempo Residencia (min)", value=3.0)
with c4:
    # Opción Especial Punto 6
    if well_type == "Pozo de Petróleo":
        use_max_api_liq = st.checkbox("Usar Max API Liq si > 600?", value=False, help="Si no se marca, se limita a 600 m3/d")
    else:
        use_max_api_liq = False # Gas wells cap at 750 strict

# --- CÁLCULOS CENTRALES ---
T_k = t_val + 273.15
P_abs_bar = p_val + 1.01325

# 1. Propiedades del Gas
Z, rho_gas_op, M = calc_peng_robinson(T_k, P_abs_bar, z_comp)
# Condiciones Standard
T_std, P_std = 288.15, 1.01325
Z_std, _, _ = calc_peng_robinson(T_std, P_std, z_comp)
# Factor Bg (Vol_Op / Vol_Std)
Bg = (P_std * T_k * Z) / (P_abs_bar * T_std * Z_std)
Factor_Std_to_Op = Bg

# 2. Propiedades del Líquido
# Punto 4: Si es Gas, fuerza 830, si es Oil, usa input (default 850)
if well_type == "Pozo de Gas":
    rho_liq = 830.0
else:
    rho_liq = st.session_state['liq_density_input']

# Calculo Geométrico Base
Area_Total = np.pi * ((D_inch*0.0254)/2)**2
Vol_Total = Area_Total * (L_ft*0.3048)

# --- GENERACIÓN DE TABLA RESUMEN (PUNTO 8) ---
st.markdown("---")
st.header("Tablas de Verificación (Escenarios de Weir Plate)")

scenarios = ["40%", "50%", "60%"]
results = []

for scen in scenarios:
    # 1. Volumen y Areas
    vol_liq_scen = VOLUMES_FIXED[EQ_KEY][scen]
    area_liq_scen = vol_liq_scen / (L_ft*0.3048)
    area_gas_scen = Area_Total - area_liq_scen
    
    # 2. Capacidad TEÓRICA API 12J
    # Gas: v = K * sqrt((rhoL-rhoG)/rhoG). K approx 0.15 m/s (con mist extractor)
    K_factor = 0.15 
    v_max_gas = K_factor * np.sqrt((rho_liq - rho_gas_op)/rho_gas_op)
    q_gas_theo_op_m3d = (v_max_gas * area_gas_scen) * 86400
    q_gas_theo_std_m3d = q_gas_theo_op_m3d / Factor_Std_to_Op
    
    # Liq: Q = V / t
    q_liq_theo_m3d = (vol_liq_scen / res_time) * 1440
    
    # 3. Capacidad REAL (Adoptada) según Reglas
    # Regla Gas
    limit_gas_std = LIMITS[EQ_KEY]["Gas_Max"]
    if well_type == "Pozo de Petróleo":
        # Punto 5: Oil Wells gas cap 200,000. Siempre usar el MENOR.
        q_gas_real_std = min(q_gas_theo_std_m3d, 200000.0)
    else:
        # Punto 3: Gas Wells cap 2,500,000. Usar calculado pero topeado.
        q_gas_real_std = min(q_gas_theo_std_m3d, 2500000.0)
    
    q_gas_real_op = q_gas_real_std * Factor_Std_to_Op

    # Regla Líquido
    limit_liq = LIMITS[EQ_KEY]["Liq_Max"] # 750 (Gas) o 600 (Oil)
    if well_type == "Pozo de Gas":
        # Punto 4: Limitar a 750
        q_liq_real = min(q_liq_theo_m3d, 750.0)
    else:
        # Punto 6: Oil Well. Limitar a 600 SALVO que usuario quiera max API
        if use_max_api_liq:
            q_liq_real = q_liq_theo_m3d
        else:
            q_liq_real = min(q_liq_theo_m3d, 600.0)
            
    # Mix Entrada (Para Ve)
    q_mix_op = q_gas_real_op + q_liq_real
    rho_mix = (q_gas_real_op*rho_gas_op + q_liq_real*rho_liq) / q_mix_op
    
    # 4. Velocidad Erosional (API 14E)
    C = 125
    # Gas Salida
    rho_g_lb = rho_gas_op * 0.062428
    Ve_gas_fts = C / np.sqrt(rho_g_lb)
    Ve_gas_ms = Ve_gas_fts * 0.3048
    
    # Líquido Salida (Usamos C=125 conservador para bifásico aunque sea liquido)
    rho_l_lb = rho_liq * 0.062428
    Ve_liq_fts = C / np.sqrt(rho_l_lb)
    Ve_liq_ms = Ve_liq_fts * 0.3048
    
    # Mix Entrada
    rho_m_lb = rho_mix * 0.062428
    Ve_mix_fts = C / np.sqrt(rho_m_lb)
    Ve_mix_ms = Ve_mix_fts * 0.3048
    
    # 5. Diámetros Requeridos (Area = Q / Ve)
    # Gas Out
    area_req_gas_m2 = (q_gas_real_op/86400) / Ve_gas_ms
    d_req_gas_mm = np.sqrt(area_req_gas_m2 * 4 / np.pi) * 1000
    
    # Liq Out
    area_req_liq_m2 = (q_liq_real/86400) / Ve_liq_ms
    d_req_liq_mm = np.sqrt(area_req_liq_m2 * 4 / np.pi) * 1000
    
    # Mix In
    area_req_mix_m2 = (q_mix_op/86400) / Ve_mix_ms
    d_req_mix_mm = np.sqrt(area_req_mix_m2 * 4 / np.pi) * 1000
    
    # 6. Guardar Datos para Tabla
    
    # Fila GAS
    results.append({
        "Escenario": f"Weir {scen}",
        "Fluido": "GAS (Salida)",
        "Caudal Teórico": f"{q_gas_theo_std_m3d:,.0f} Sm3/d",
        "Caudal Real/Adoptado": f"{q_gas_real_std:,.0f} Sm3/d",
        "Condiciones": f"{p_val} bar / {t_val} °C",
        "Ve Calc (m/s)": f"{Ve_gas_ms:.2f}",
        "ID Req (mm)": f"{d_req_gas_mm:.1f}",
        "Brida Min #600": get_closest_flange(d_req_gas_mm),
        "Info Extra": f"Op: {q_gas_real_op:.1f} m3/d"
    })
    
    # Fila LIQ
    results.append({
        "Escenario": f"Weir {scen}",
        "Fluido": "LIQUIDO (Salida)",
        "Caudal Teórico": f"{q_liq_theo_m3d:,.0f} m3/d",
        "Caudal Real/Adoptado": f"{q_liq_real:,.0f} m3/d",
        "Condiciones": f"{p_val} bar / {t_val} °C",
        "Ve Calc (m/s)": f"{Ve_liq_ms:.2f}",
        "ID Req (mm)": f"{d_req_liq_mm:.1f}",
        "Brida Min #600": get_closest_flange(d_req_liq_mm),
        "Info Extra": f"Dens: {rho_liq} kg/m3"
    })

    # Fila MIX (Entrada) - Opcional pero útil para verificar entrada
    results.append({
        "Escenario": f"Weir {scen}",
        "Fluido": "ENTRADA (Mix)",
        "Caudal Teórico": "-",
        "Caudal Real/Adoptado": f"{q_mix_op:,.0f} m3/d (Act)",
        "Condiciones": f"{p_val} bar / {t_val} °C",
        "Ve Calc (m/s)": f"{Ve_mix_ms:.2f}",
        "ID Req (mm)": f"{d_req_mix_mm:.1f}",
        "Brida Min #600": get_closest_flange(d_req_mix_mm),
        "Info Extra": "-"
    })

df_res = pd.DataFrame(results)
st.dataframe(df_res, use_container_width=True)

# --- VERIFICACION DETALLADA DE BOQUILLAS (Punto 1) ---
st.markdown("---")
st.header("Verificación Detallada de Cañerías (Sch 40/80)")
st.info("Esta tabla verifica velocidades erosionales para el ESCENARIO ACTUAL de operación seleccionado abajo.")

# Selector temporal para ver el detalle de un caso específico
scen_sel = st.selectbox("Seleccionar Escenario para detalle:", scenarios)
q_gas_sel = float(df_res[(df_res['Escenario']==f"Weir {scen_sel}") & (df_res['Fluido']=="GAS (Salida)")]["Info Extra"].values[0].split()[1])
q_liq_sel = float(df_res[(df_res['Escenario']==f"Weir {scen_sel}") & (df_res['Fluido']=="LIQUIDO (Salida)")]["Caudal Real/Adoptado"].values[0].split()[0].replace(",",""))
q_mix_sel = q_gas_sel + q_liq_sel

# Recalcular Ve para este punto (ya calculados antes, recuperamos valores limpios)
# Simplificación: recalculamos rápido para mostrar la tabla dinámica
# ... (usamos las densidades globales calculadas arriba)

sizes_check = ["4 inch", "6 inch", "8 inch", "10 inch"]
pipe_rows = []

for sz in sizes_check:
    for sch in ["Sch40", "Sch80"]:
        id_in = PIPES_DB[sz][sch]
        area_m2 = np.pi * ((id_in*0.0254)/2)**2
        
        # Gas Out
        v_gas = (q_gas_sel/86400) / area_m2
        ve_gas = 125 / np.sqrt(rho_gas_op * 0.062428) * 0.3048
        stt_gas = "✅" if v_gas < ve_gas else "❌"
        
        # Mix In (usando Q mix op)
        # Densidad mix aproximada para este punto
        rho_m_iter = (q_gas_sel*rho_gas_op + q_liq_sel*rho_liq) / q_mix_sel
        ve_mix = 125 / np.sqrt(rho_m_iter * 0.062428) * 0.3048
        v_mix = (q_mix_sel/86400) / area_m2
        stt_mix = "✅" if v_mix < ve_mix else "❌"

        pipe_rows.append([
            sz, sch, f"{id_in}\"", 
            f"{v_gas:.1f} / {ve_gas:.1f} {stt_gas}",
            f"{v_mix:.1f} / {ve_mix:.1f} {stt_mix}"
        ])

df_pipes = pd.DataFrame(pipe_rows, columns=["DN", "Sch", "ID", "Vel Gas Real / Ve (m/s)", "Vel Mix In / Ve (m/s)"])
st.dataframe(df_pipes, use_container_width=True)

# Avisos de lógica aplicada
if well_type == "Pozo de Gas":
    st.warning(f"Nota: Se asumió densidad de líquido = 830 kg/m3 (Agua) y Caudal Max Liq = 750 m3/d.")
    if df_res['Caudal Teórico'].str.contains('Sm3/d').any():
         pass # Placeholder logic check
else:
    st.warning(f"Nota: Se limitó Caudal Gas a 200,000 Sm3/d (Menor valor entre API y Límite).")

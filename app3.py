import streamlit as st
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Verificación SEP-10/SEP-30", layout="wide", page_icon="⚡")

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

# Composiciones Fijas
COMP_GAS = {"N2":0.204,"CO2":0.748,"CH4":94.87,"C2H6":3.721,"C3H8":0.376,"iC4H10":0.039,"nC4H10":0.028,"iC5H12":0.0,"nC5H12":0.0,"C6H14":0.015,"C7H16":0.0,"C8H18":0.0,"C9H20+":0.0,"O2":0.0,"H2O":0.0}
COMP_PETROLEO = {"N2":1.94,"CO2":0.138,"CH4":65.5,"C2H6":16.64,"C3H8":10.0,"iC4H10":0.61,"nC4H10":2.61,"iC5H12":0.71,"nC5H12":0.96,"C6H14":0.57,"C7H16":0.31,"C8H18":0.03,"C9H20+":0.0,"O2":0.0,"H2O":0.0}

PIPES_DB = {
    "3 inch": {"Sch40": 3.068, "Sch80": 2.900},
    "4 inch": {"Sch40": 4.026, "Sch80": 3.826},
    "6 inch": {"Sch40": 6.065, "Sch80": 5.761},
    "8 inch": {"Sch40": 7.981, "Sch80": 7.625},
    "10 inch": {"Sch40": 10.02, "Sch80": 9.562},
    "12 inch": {"Sch40": 12.00, "Sch80": 11.374},
}

# Límites (Hard Constraints)
LIMITS = {
    "SEP-10": {"Gas_Max": 2500000.0, "Liq_Max": 750.0},
    "SEP-30": {"Gas_Max": 200000.0, "Liq_Max": 600.0}
}

# Volúmenes fijos (m3)
VOLUMES_FIXED = {
    "SEP-10": {"40%": 1.47, "50%": 1.967, "60%": 2.465},
    "SEP-30": {"40%": 0.827, "50%": 1.107, "60%": 1.386}
}

# --- 3. FUNCIONES ---
def calc_peng_robinson(T_k, P_bar, composition_mole_frac):
    R = 8.314462618
    P_pa = P_bar * 1e5
    M_mix = 0
    params = {}
    
    # Normalizar si no viene normalizado
    total_mol = sum(composition_mole_frac.values())
    if total_mol == 0: return 1, 0, 0
    
    for comp, val in composition_mole_frac.items():
        y = val / total_mol # asegurar fraccion 0-1
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
    stds = [(3, 2.9), (4, 3.826), (6, 5.761), (8, 7.625), (10, 9.562), (12, 11.37)]
    id_req_inch = id_req_mm / 25.4
    for dn, bore in stds:
        if bore >= id_req_inch: return f"{dn}\""
    return ">12\""

# --- 4. INTERFAZ ---
st.title("🏭 Verificación de Equipos: SEP-10 & SEP-30")

# --- SELECCION ---
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    equip_sel = st.selectbox("Equipo", ["SEP-10 (48x15)", "SEP-30 (36x15)"])
with col_sel2:
    well_type = st.radio("Tipo de Pozo", ["Pozo de Gas", "Pozo de Petróleo"], horizontal=True)

# Lógica de Selección Automática
if "SEP-10" in equip_sel:
    EQ_KEY = "SEP-10"
    D_inch, L_ft = 48.0, 15.0
else:
    EQ_KEY = "SEP-30"
    D_inch, L_ft = 36.0, 15.0

# Asignación automática de Composición y Densidad Líquido
if well_type == "Pozo de Gas":
    comp_actual = COMP_GAS
    rho_liq = 830.0 # Agua predominante
    label_liq = "Agua (830 kg/m3)"
else:
    comp_actual = COMP_PETROLEO
    rho_liq = 850.0 # Petróleo promedio
    label_liq = "Petróleo (850 kg/m3)"

# Mostrar configuración activa (solo lectura informativa)
st.info(f"**Configuración Activa:** Cromatografía: {well_type} | Densidad Líquido: {rho_liq} kg/m3 ({label_liq})")

# --- CONDICIONES OPERATIVAS ---
st.markdown("---")
st.header("Condiciones Operativas")
c1, c2, c3, c4 = st.columns(4)
with c1:
    t_val = st.number_input("Temperatura (°C)", value=45.0)
with c2:
    p_val = st.number_input("Presión (bar g)", value=90.0)
with c3:
    res_time = st.number_input("Tiempo Residencia (min)", value=3.0)
with c4:
    if well_type == "Pozo de Petróleo":
        use_max_api_liq = st.checkbox("Usar Max API Liq si > 600?", value=False)
    else:
        use_max_api_liq = False

# --- CÁLCULOS CENTRALES ---
T_k = t_val + 273.15
P_abs_bar = p_val + 1.01325

# 1. Propiedades del Gas
Z, rho_gas_op, M = calc_peng_robinson(T_k, P_abs_bar, comp_actual)
# Condiciones Standard (15C, 1atm para calculo Bg)
T_std, P_std = 288.15, 1.01325
Z_std, _, _ = calc_peng_robinson(T_std, P_std, comp_actual)
# Factor Bg (Vol_Std * Bg = Vol_Op) => Bg = (Pstd/Pop)*(Top/Tstd)*(Zop/Zstd)
# Vol_Op = Vol_Std * (Pstd/Pop) ...
Factor_Std_to_Op = (P_std/P_abs_bar) * (T_k/T_std) * (Z/Z_std)

Area_Total = np.pi * ((D_inch*0.0254)/2)**2

# --- GENERACIÓN DE TABLA DETALLADA (PUNTO 8) ---
st.markdown("---")
st.header("Resultados Detallados por Escenario")

scenarios = ["40%", "50%", "60%"]
table_data = []

for scen in scenarios:
    # A. Volúmenes
    vol_liq_scen = VOLUMES_FIXED[EQ_KEY][scen]
    area_liq_scen = vol_liq_scen / (L_ft*0.3048)
    area_gas_scen = Area_Total - area_liq_scen
    
    # B. Capacidad TEÓRICA API 12J
    # Gas Max Teórico
    K_factor = 0.15 # m/s (approx 0.5 ft/s)
    v_max_gas = K_factor * np.sqrt((rho_liq - rho_gas_op)/rho_gas_op)
    q_gas_theo_op_m3d = (v_max_gas * area_gas_scen) * 86400
    q_gas_theo_std_m3d = q_gas_theo_op_m3d / Factor_Std_to_Op
    
    # Liq Max Teórico
    q_liq_theo_m3d = (vol_liq_scen / res_time) * 1440.0
    
    # C. Capacidad REAL (Adoptada por Reglas)
    # Reglas Gas
    limit_gas_std = LIMITS[EQ_KEY]["Gas_Max"]
    if well_type == "Pozo de Petróleo":
        # Siempre usar el MENOR entre Teórico y 200k
        q_gas_real_std = min(q_gas_theo_std_m3d, 200000.0)
    else:
        # Gas Wells: Calcular pero topear en 2.5MM
        q_gas_real_std = min(q_gas_theo_std_m3d, 2500000.0)
    
    q_gas_real_op = q_gas_real_std * Factor_Std_to_Op
    
    # Reglas Líquido
    if well_type == "Pozo de Gas":
        # Limitar a 750 (mayormente agua)
        q_liq_real = min(q_liq_theo_m3d, 750.0)
        q_oil_out = 0 # Asumimos agua
        q_water_out = q_liq_real
    else:
        # Pozo Petróleo
        if use_max_api_liq:
             q_liq_real = q_liq_theo_m3d
        else:
             q_liq_real = min(q_liq_theo_m3d, 600.0)
        # Asumimos todo petróleo para el output de tabla (GOR manda gas, Liq es Oil)
        q_oil_out = q_liq_real
        q_water_out = 0
        
    # Caudal de Ingreso (Mix)
    q_in_mix_op = q_gas_real_op + q_liq_real
    rho_mix = (q_gas_real_op*rho_gas_op + q_liq_real*rho_liq) / q_in_mix_op
    
    # D. Velocidad Erosional y Boquillas
    C = 125
    # Gas Out
    rho_g_lb = rho_gas_op * 0.062428
    Ve_gas_fts = C / np.sqrt(rho_g_lb)
    Ve_gas_ms = Ve_gas_fts * 0.3048
    d_req_gas_mm = np.sqrt(((q_gas_real_op/86400)/Ve_gas_ms)*4/np.pi)*1000
    
    # Liq Out
    rho_l_lb = rho_liq * 0.062428
    Ve_liq_fts = C / np.sqrt(rho_l_lb)
    Ve_liq_ms = Ve_liq_fts * 0.3048
    d_req_liq_mm = np.sqrt(((q_liq_real/86400)/Ve_liq_ms)*4/np.pi)*1000
    
    # In Mix
    rho_m_lb = rho_mix * 0.062428
    Ve_mix_fts = C / np.sqrt(rho_m_lb)
    Ve_mix_ms = Ve_mix_fts * 0.3048
    d_req_mix_mm = np.sqrt(((q_in_mix_op/86400)/Ve_mix_ms)*4/np.pi)*1000
    
    # E. Verificación Cañerías (Sch40/80) - Boquilla Entrada
    # Tomamos referencia 4" para SEP-30 y 6" para SEP-10 como base para el check booleano
    # Pero el usuario pidió tabla, así que resumimos el check del diámetro nominal
    base_in_sz = "6 inch" if "SEP-10" in EQ_KEY else "4 inch"
    id_check = PIPES_DB[base_in_sz]["Sch80"]
    v_check = (q_in_mix_op/86400) / (np.pi*(id_check*0.0254/2)**2)
    check_status = "OK" if v_check < Ve_mix_ms else "FAIL"

    # Construcción de Filas para la tabla Grande
    # Columna 1 datos
    col1_gas_theo_std = q_gas_theo_std_m3d
    col1_gas_real_std = q_gas_real_std
    col1_gas_theo_act = q_gas_theo_op_m3d
    col1_gas_real_act = q_gas_real_op
    col1_in_theo = "-" # No aplica un "teorico" de entrada, depende del pozo
    col1_in_real = q_in_mix_op
    
    # Agregar fila al dataset
    # Estructura: Escenario | Fluido | Q Teo | Q Real | Pres/Temp | Ve | D Req | Brida | Verif Pipe
    
    # 1. GAS OUT
    table_data.append({
        "Escenario": f"Weir {scen}", "Fluido": "GAS OUT",
        "Q Teórico (Std)": f"{col1_gas_theo_std:,.0f}", "Q Real (Std)": f"{col1_gas_real_std:,.0f}",
        "Q Real (Act)": f"{col1_gas_real_act:,.0f}",
        "P/T": f"{p_val}/{t_val}", "Ve (m/s)": f"{Ve_gas_ms:.2f}",
        "D Req (mm)": f"{d_req_gas_mm:.1f}", "Brida #600": get_closest_flange(d_req_gas_mm),
        "Verif": f"Ve Gas: {Ve_gas_ms:.1f} m/s"
    })
    
    # 2. LIQ OUT
    liq_label = "AGUA OUT" if q_water_out > 0 else "OIL OUT"
    table_data.append({
        "Escenario": f"Weir {scen}", "Fluido": liq_label,
        "Q Teórico (Std)": "-", "Q Real (Std)": "-",
        "Q Real (Act)": f"{q_liq_real:,.0f}",
        "P/T": f"{p_val}/{t_val}", "Ve (m/s)": f"{Ve_liq_ms:.2f}",
        "D Req (mm)": f"{d_req_liq_mm:.1f}", "Brida #600": get_closest_flange(d_req_liq_mm),
        "Verif": "-"
    })
    
    # 3. INLET
    table_data.append({
        "Escenario": f"Weir {scen}", "Fluido": "ENTRADA MIX",
        "Q Teórico (Std)": "-", "Q Real (Std)": "-",
        "Q Real (Act)": f"{col1_in_real:,.0f}",
        "P/T": f"{p_val}/{t_val}", "Ve (m/s)": f"{Ve_mix_ms:.2f}",
        "D Req (mm)": f"{d_req_mix_mm:.1f}", "Brida #600": get_closest_flange(d_req_mix_mm),
        "Verif": f"{base_in_sz} Sch80: {check_status}"
    })

# Renderizar Tabla Principal
df_main = pd.DataFrame(table_data)
st.dataframe(df_main, use_container_width=True)

# --- VERIFICACIÓN DE BOQUILLAS (Tabla Detallada) ---
st.markdown("---")
st.header("Verificación de Boquillas (Sch 40 vs Sch 80)")
st.info("Verificación de Velocidad Erosional para los Caudales REALES calculados arriba.")

# Selector de escenario para detalle
sel_scen = st.selectbox("Seleccionar Escenario Weir para ver detalle:", scenarios)

# Recuperar caudales del escenario seleccionado
row_gas = df_main[(df_main['Escenario']==f"Weir {sel_scen}") & (df_main['Fluido']=="GAS OUT")].iloc[0]
q_gas_act_sel = float(row_gas['Q Real (Act)'].replace(",",""))

row_in = df_main[(df_main['Escenario']==f"Weir {sel_scen}") & (df_main['Fluido']=="ENTRADA MIX")].iloc[0]
q_mix_act_sel = float(row_in['Q Real (Act)'].replace(",",""))

# Recalcular Ve específicas
# Gas Ve
ve_gas_ms_val = float(row_gas['Ve (m/s)'])
# Mix Ve
ve_mix_ms_val = float(row_in['Ve (m/s)'])

pipe_rows = []
sizes_to_check = ["4 inch", "6 inch", "8 inch", "10 inch"]

for sz in sizes_to_check:
    for sch in ["Sch40", "Sch80"]:
        id_in = PIPES_DB[sz][sch]
        area_m2 = np.pi * ((id_in*0.0254)/2)**2
        
        # Gas Out Check
        v_gas = (q_gas_act_sel/86400) / area_m2
        stt_gas = "✅" if v_gas < ve_gas_ms_val else "❌"
        
        # Mix In Check
        v_mix = (q_mix_act_sel/86400) / area_m2
        stt_mix = "✅" if v_mix < ve_mix_ms_val else "❌"
        
        pipe_rows.append({
            "DN": sz, "Sch": sch, "ID (pulg)": id_in,
            "Vel Gas (m/s)": f"{v_gas:.1f}", 
            "Ve Gas Limite": f"{ve_gas_ms_val:.1f}",
            "Estado Gas": stt_gas,
            "Vel Mix (m/s)": f"{v_mix:.1f}",
            "Ve Mix Limite": f"{ve_mix_ms_val:.1f}",
            "Estado Mix": stt_mix
        })

st.dataframe(pd.DataFrame(pipe_rows), use_container_width=True)

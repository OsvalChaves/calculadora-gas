import streamlit as st
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Vessel Check AI", layout="wide", page_icon="⚡")

# --- 2. BASE DE DATOS COMPONENTES ---
COMPONENTS_DB = {
    "N2":      {"Tc": 126.2,  "Pc": 33.9,  "w": 0.037, "M": 28.01},
    "CO2":     {"Tc": 304.1,  "Pc": 73.8,  "w": 0.225, "M": 44.01},
    "H2S":     {"Tc": 373.2,  "Pc": 89.4,  "w": 0.099, "M": 34.08},
    "CH4":     {"Tc": 190.6,  "Pc": 46.0,  "w": 0.011, "M": 16.04},
    "C2H6":    {"Tc": 305.4,  "Pc": 48.8,  "w": 0.099, "M": 30.07},
    "C3H8":    {"Tc": 369.8,  "Pc": 42.5,  "w": 0.152, "M": 44.10},
    "iC4H10":  {"Tc": 408.2,  "Pc": 36.5,  "w": 0.183, "M": 58.12},
    "nC4H10":  {"Tc": 425.2,  "Pc": 38.0,  "w": 0.199, "M": 58.12},
    "iC5H12":  {"Tc": 460.4,  "Pc": 33.8,  "w": 0.227, "M": 72.15},
    "nC5H12":  {"Tc": 469.7,  "Pc": 33.7,  "w": 0.251, "M": 72.15},
    "C6H14":   {"Tc": 507.5,  "Pc": 30.1,  "w": 0.299, "M": 86.18},
    "C7H16":   {"Tc": 540.2,  "Pc": 27.4,  "w": 0.349, "M": 100.21},
    "C8H18":   {"Tc": 568.8,  "Pc": 24.9,  "w": 0.398, "M": 114.23},
    "C9H20+":  {"Tc": 594.6,  "Pc": 22.9,  "w": 0.445, "M": 128.26},
    "O2":      {"Tc": 154.6,  "Pc": 50.4,  "w": 0.022, "M": 31.99},
    "H2O":     {"Tc": 647.1,  "Pc": 220.6, "w": 0.344, "M": 18.015}
}

# --- 3. DATA TUBERIAS ---
# Diccionario simplificado con ID en pulgadas para Sch40/80
PIPES_DB = {
    "3 inch":  {"Sch40": 3.068, "Sch80": 2.900},
    "4 inch":  {"Sch40": 4.026, "Sch80": 3.826},
    "6 inch":  {"Sch40": 6.065, "Sch80": 5.761},
    "8 inch":  {"Sch40": 7.981, "Sch80": 7.625},
    "10 inch": {"Sch40": 10.02, "Sch80": 9.562},
    "12 inch": {"Sch40": 12.00, "Sch80": 11.374}, 
}

# --- 4. FUNCIONES ---
def calc_peng_robinson(T_k, P_bar, composition_mole_frac):
    R = 8.314462618
    P_pa = P_bar * 1e5
    Tc_mix, Pc_mix, w_mix, M_mix = 0, 0, 0, 0
    a_mix, b_mix = 0, 0
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
    Z = max(real_roots)
    rho_kg_m3 = (P_pa * (M_mix / 1000)) / (Z * R * T_k)
    return Z, rho_kg_m3, M_mix

# Presets y Gestión de Estado
PRESET_PETROLEO = {"N2":1.94,"CO2":0.138,"CH4":65.5,"C2H6":16.64,"C3H8":10.0,"iC4H10":0.61,"nC4H10":2.61,"iC5H12":0.71,"nC5H12":0.96,"C6H14":0.57,"C7H16":0.31,"C8H18":0.03,"C9H20+":0.0,"O2":0.0,"H2O":0.0}
PRESET_GAS = {"N2":0.204,"CO2":0.748,"CH4":94.87,"C2H6":3.721,"C3H8":0.376,"iC4H10":0.039,"nC4H10":0.028,"iC5H12":0.0,"nC5H12":0.0,"C6H14":0.015,"C7H16":0.0,"C8H18":0.0,"C9H20+":0.0,"O2":0.0,"H2O":0.0}

if 'df_comp' not in st.session_state:
    data = {"Componente": list(COMPONENTS_DB.keys()), "% Molar": [0.0]*len(COMPONENTS_DB)}
    st.session_state.df_comp = pd.DataFrame(data)
    st.session_state.df_comp.loc[st.session_state.df_comp['Componente']=='CH4', '% Molar'] = 100.0

if 'liq_density_input' not in st.session_state:
    st.session_state['liq_density_input'] = 850.0

def load_petroleo():
    for i, row in st.session_state.df_comp.iterrows():
        st.session_state.df_comp.at[i, '% Molar'] = PRESET_PETROLEO.get(row['Componente'], 0.0)
    st.session_state['liq_density_input'] = 850.0

def load_gas():
    for i, row in st.session_state.df_comp.iterrows():
        st.session_state.df_comp.at[i, '% Molar'] = PRESET_GAS.get(row['Componente'], 0.0)
    st.session_state['liq_density_input'] = 550.0

# --- INTERFAZ ---
st.title("🏭 Verificación de Equipos: 48x15 & 36x15")

# --- SELECCIÓN DE EQUIPO ---
st.sidebar.markdown("---")
st.sidebar.header("Selección de Equipo")
equip_type = st.sidebar.selectbox("Separador a Verificar", ["48x15 (Gas/Flowback)", "36x15 (Petróleo/Testing)"])

# Definir dimensiones según selección
if "48x15" in equip_type:
    # 48 inch ID, 15 ft Length
    V_ID_inch = 48.0
    V_L_ft = 15.0
    # Configuración de tuberías/boquillas del mail
    PIPE_IN_SZ = "6 inch"
    PIPE_OUT_SZ = "6 inch"
    NOZ_IN_OPT = ["8 inch", "10 inch"]
    NOZ_OUT_OPT = ["8 inch", "10 inch"]
    # Targets comerciales
    TARGET_GAS = 2500000.0 # 2.5 MM Sm3/d
    TARGET_LIQ = 750.0     # 750 m3/d
    TARGET_EPF_LIQ = 1000.0 # Oil EPF verification
else:
    # 36 inch ID, 15 ft Length
    V_ID_inch = 36.0
    V_L_ft = 15.0
    # Configuración de tuberías/boquillas del mail
    PIPE_IN_SZ = "4 inch"
    PIPE_OUT_SZ = "4 inch"
    NOZ_IN_OPT = ["6 inch"]
    NOZ_OUT_OPT = ["6 inch"]
    # Targets comerciales
    TARGET_GAS = 200000.0 # 200k Sm3/d (Primary)
    TARGET_GAS_ALT = 850000.0 # 850k Sm3/d (YPF Gas Check)
    TARGET_LIQ = 600.0     # 600 m3/d

# Geometría Interna
V_ID_m = V_ID_inch * 0.0254
V_L_m = V_L_ft * 0.3048
V_TOTAL_m3 = (np.pi * (V_ID_m/2)**2) * V_L_m

# --- 1. COMPOSICION ---
st.sidebar.markdown("---")
st.sidebar.header("1. Composición del Fluido")
c1, c2 = st.sidebar.columns(2)
c1.button("Pozo Petróleo", on_click=load_petroleo, use_container_width=True)
c2.button("Pozo Gas", on_click=load_gas, use_container_width=True)

edited_df = st.sidebar.data_editor(st.session_state.df_comp, hide_index=True, height=400, use_container_width=True)
st.session_state.df_comp = edited_df
mole_fractions = {r['Componente']: r['% Molar']/100 for i, r in edited_df.iterrows()}

# --- 2. CONDICIONES ---
st.header("2. Condiciones del Ensayo (Inputs)")
col_g1, col_g2, col_g3, col_g4 = st.columns(4)

with col_g1:
    t_u = st.selectbox("Unidad T", ["°C", "°F"])
    t_val = st.number_input("Temp", 45.0)
with col_g2:
    p_u = st.selectbox("Unidad P", ["bar", "kg/cm2", "psi"])
    p_val = st.number_input("Presión (Gauge)", 90.0) # Default alto como pide el mail
with col_g3:
    cond_q = st.selectbox("Ref Caudal Gas", ["Std (60F, 14.7p)", "Operativo"])
with col_g4:
    if "Std" in cond_q: u_q = st.selectbox("Unidad Gas", ["Sm3/d", "MMSCFD"])
    else: u_q = st.selectbox("Unidad Gas", ["m3/d", "m3/h"])
    q_gas_val = st.number_input("Flujo Gas", 200000.0)

# --- 3. FASE LIQUIDA ---
st.markdown("---")
st.header("3. Configuración de Líquido y Separación")
col_l1, col_l2, col_l3, col_l4 = st.columns(4)

with col_l1:
    res_time = st.number_input("Tiempo Residencia (min)", 1.0, 60.0, 3.0)
with col_l2:
    vol_liq_val = st.number_input("Volumen Separación (m3)", 0.0, V_TOTAL_m3, V_TOTAL_m3*0.4, help=f"Volumen total del vessel es {V_TOTAL_m3:.2f} m3")
with col_l3:
    rho_liq_val = st.number_input("Densidad Líquido (kg/m3)", min_value=1.0, max_value=2000.0, key="liq_density_input")

with col_l4:
    # Capacidad Líquido según Input
    q_liq_m3d_cap = (vol_liq_val / res_time) * 1440.0
    st.metric("Capacidad Líquido (Q = V/t)", f"{q_liq_m3d_cap:.0f} m3/d")
    
    # Check vs Total Volume
    liq_level_frac = vol_liq_val / V_TOTAL_m3
    st.progress(min(liq_level_frac, 1.0), text=f"Llenado Volumen: {liq_level_frac*100:.1f}%")

# --- CALCULOS TERMO ---
T_k = t_val + 273.15 if t_u == "°C" else (t_val - 32)*5/9 + 273.15
if p_u == "bar": P_bar = p_val + 1.013
elif p_u == "kg/cm2": P_bar = (p_val * 0.980665) + 1.013
else: P_bar = (p_val+14.7)/14.5

Z, rho_gas, M = calc_peng_robinson(T_k, P_bar, mole_fractions)

# Conversion Caudales
T_std, P_std = 288.71, 1.01353 # 60F
Z_std, _, _ = calc_peng_robinson(T_std, P_std, mole_fractions)
factor = (P_std/P_bar) * (T_k/T_std) * (Z/Z_std)

q_gas_act_m3d = 0.0
q_gas_std_m3d = 0.0

if "Std" in cond_q:
    qs = q_gas_val
    if u_q=="MMSCFD": qs = q_gas_val * 1e6 * 0.0283168
    q_gas_std_m3d = qs
    q_gas_act_m3d = qs * factor
else:
    qa = q_gas_val
    if u_q=="m3/h": qa = q_gas_val * 24
    q_gas_act_m3d = qa
    q_gas_std_m3d = qa / factor

# Mezcla Entrada (Asumiendo Q_liq = Capacidad seteada por input volumen)
q_liq_act_m3d = q_liq_m3d_cap 
q_total_act_m3d = q_gas_act_m3d + q_liq_act_m3d
q_total_m3s = q_total_act_m3d / 86400

rho_liq_kgm3 = rho_liq_val
mass_gas = q_gas_act_m3d * rho_gas
mass_liq = q_liq_act_m3d * rho_liq_kgm3
rho_mix = (mass_gas + mass_liq) / q_total_act_m3d
rho_mix_lb = rho_mix * 0.062428
rho_gas_lb = rho_gas * 0.062428

# API 14E Ve
C_erosion = 125
ve_mix_fts = C_erosion / np.sqrt(rho_mix_lb)
ve_gas_fts = C_erosion / np.sqrt(rho_gas_lb)

# --- RESULTADOS DE VERIFICACIÓN ---
st.markdown("---")
st.header(f"4. Verificación del Vessel {equip_type}")

# A. CAPACIDAD DE GAS (API 12J Horizontal Theory)
# Q_gas_max = K * A_gas * sqrt(...)
# A_gas = Area Total - Area Liq. Aproximamos Area Liq proporcional al Volumen (simplificacion valida para V=A*L)
Area_total_m2 = np.pi * (V_ID_m/2)**2
Area_liq_m2 = Area_total_m2 * (vol_liq_val / V_TOTAL_m3)
Area_gas_m2 = Area_total_m2 - Area_liq_m2

if Area_gas_m2 <= 0:
    q_gas_max_m3d = 0
    st.error("¡El separador está inundado! Volumen Líquido >= Volumen Total")
else:
    # API 12J K value (approx 0.5 ft/s for 10-15ft vessels with mist extractor)
    K_val_ms = 0.15 # ~0.5 ft/s
    v_crit_ms = K_val_ms * np.sqrt((rho_liq_kgm3 - rho_gas)/rho_gas)
    q_gas_max_m3s = v_crit_ms * Area_gas_m2
    q_gas_max_m3d = q_gas_max_m3s * 86400

# Convertir capacidad maxima a Standard para comparar
q_gas_max_std = q_gas_max_m3d / factor

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.subheader("Capacidad Teórica (Condiciones Actuales)")
    st.write(f"Con Nivel de Líquido al **{(vol_liq_val/V_TOTAL_m3)*100:.1f}%**:")
    
    # Gas
    delta_gas = q_gas_std_m3d / q_gas_max_std * 100
    st.metric("Capacidad Máx Gas (Sm3/d)", f"{q_gas_max_std:,.0f}", help="Basado en API 12J (v_crit)")
    st.progress(min(delta_gas/100, 1.0), text=f"Uso Gas: {delta_gas:.1f}%")
    
    # Líquido (Ya calculado arriba)
    st.metric("Capacidad Máx Líquido (m3/d)", f"{q_liq_m3d_cap:.0f}")

with col_res2:
    st.subheader("Verificación de Objetivos (Mail)")
    
    # Logica especifica 48x15
    if "48x15" in equip_type:
        # Check Gas Primario
        st.write("**Modo Gas / Flowback:**")
        chk1 = "✅" if q_gas_max_std >= TARGET_GAS else "⚠️"
        st.write(f"{chk1} Objetivo Gas ({TARGET_GAS:,.0f} Sm3/d)")
        
        chk2 = "✅" if q_liq_m3d_cap >= TARGET_LIQ else "⚠️"
        st.write(f"{chk2} Objetivo Liq ({TARGET_LIQ:.0f} m3/d)")
        
        st.write("**Modo Oil EPF (Verificación):**")
        chk3 = "✅" if q_liq_m3d_cap >= TARGET_EPF_LIQ else "⚠️"
        st.write(f"{chk3} Objetivo Liq EPF ({TARGET_EPF_LIQ:.0f} m3/d)")
        
    # Logica especifica 36x15
    else:
        st.write("**Modo Petróleo:**")
        chk1 = "✅" if q_gas_max_std >= TARGET_GAS else "⚠️"
        st.write(f"{chk1} Objetivo Gas ({TARGET_GAS:,.0f} Sm3/d)")
        
        chk2 = "✅" if q_liq_m3d_cap >= TARGET_LIQ else "⚠️"
        st.write(f"{chk2} Objetivo Liq ({TARGET_LIQ:.0f} m3/d)")
        
        st.write("**Modo Gas (Contrato YPF):**")
        chk3 = "✅" if q_gas_max_std >= TARGET_GAS_ALT else "❌"
        st.write(f"{chk3} Objetivo Gas YPF ({TARGET_GAS_ALT:,.0f} Sm3/d)")
        if q_gas_max_std < TARGET_GAS_ALT:
            st.caption("Aviso: El vessel no da la capacidad de gas requerida con este nivel de líquido.")

# --- 5. BOQUILLAS Y CAÑERÍAS ---
st.markdown("---")
st.header("5. Verificación de Boquillas y Cañerías")

c_in, c_out = st.columns(2)

# --- ENTRADA ---
with c_in:
    st.subheader("Entrada (Mix)")
    st.caption(f"Q_mix_actual = {q_total_act_m3d:.1f} m3/d")
    
    # 1. Cañería de Entrada (Fija según equipo)
    st.markdown(f"**Cañería de Entrada ({PIPE_IN_SZ})**")
    id_pipe = PIPES_DB[PIPE_IN_SZ]["Sch40"] # Asumimos Sch40 para la cañeria
    a_pipe = np.pi * (id_pipe/24)**2
    v_pipe = (q_total_m3s * 35.315) / a_pipe
    
    stt_pipe = "✅ OK" if v_pipe < ve_mix_fts else "❌ EROSIÓN"
    st.write(f"Velocidad: {v_pipe:.1f} ft/s (Ve: {ve_mix_fts:.1f}) -> {stt_pipe}")
    
    # 2. Boquilla Entrada (Seleccionable)
    st.markdown("**Boquilla de Entrada**")
    noz_select = st.selectbox("Seleccionar Boquilla In", NOZ_IN_OPT)
    id_noz = PIPES_DB[noz_select]["Sch80"] # Boquilla suele ser Sch80 o más
    a_noz = np.pi * (id_noz/24)**2
    v_noz = (q_total_m3s * 35.315) / a_noz
    mom_noz = rho_mix_lb * v_noz**2
    
    st.write(f"Momentum: **{mom_noz:.0f}** lb/ft s2")
    if mom_noz > 3000: st.error("❌ Momentum Excesivo (>3000)")
    else: st.success("✅ Momentum OK")

# --- SALIDA GAS ---
with c_out:
    st.subheader("Salida Gas")
    st.caption(f"Q_gas_actual = {q_gas_act_m3d:.1f} m3/d")
    
    # 1. Cañería Salida (Fija según equipo)
    st.markdown(f"**Cañería Salida ({PIPE_OUT_SZ})**")
    id_pipe_g = PIPES_DB[PIPE_OUT_SZ]["Sch40"]
    a_pipe_g = np.pi * (id_pipe_g/24)**2
    v_pipe_g = (q_gas_act_m3d/86400 * 35.315) / a_pipe_g
    
    stt_pipe_g = "✅ OK" if v_pipe_g < ve_gas_fts else "❌ EROSIÓN"
    st.write(f"Velocidad: {v_pipe_g:.1f} ft/s (Ve: {ve_gas_fts:.1f}) -> {stt_pipe_g}")
    
    # 2. Boquilla Salida (Seleccionable)
    st.markdown("**Boquilla Salida**")
    noz_out_select = st.selectbox("Seleccionar Boquilla Out", NOZ_OUT_OPT)
    id_noz_g = PIPES_DB[noz_out_select]["Sch80"]
    a_noz_g = np.pi * (id_noz_g/24)**2
    v_noz_g = (q_gas_act_m3d/86400 * 35.315) / a_noz_g
    mom_noz_g = rho_gas_lb * v_noz_g**2
    
    st.write(f"Momentum: **{mom_noz_g:.0f}** lb/ft s2")
    if mom_noz_g > 4000: st.error("❌ Momentum Excesivo (>4000)")
    else: st.success("✅ Momentum OK")
    
    # Check Bypass
    if "48x15" in equip_type:
        st.info("Nota: Bypass o SDV de 4\" requerido por mail.")

st.markdown("---")
st.caption("Nota: Cálculo de capacidad de gas basado en teoría de separación horizontal (API 12J) con K=0.15 m/s (~0.5 ft/s).")

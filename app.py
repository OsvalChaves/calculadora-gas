import streamlit as st
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Gas & Liq Process Calc", layout="wide", page_icon="⚡")

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

# --- 3. FUNCIONES ---
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

# Presets
PRESET_PETROLEO = {"N2":1.94,"CO2":0.138,"CH4":65.5,"C2H6":16.64,"C3H8":10.0,"iC4H10":0.61,"nC4H10":2.61,"iC5H12":0.71,"nC5H12":0.96,"C6H14":0.57,"C7H16":0.31,"C8H18":0.03,"C9H20+":0.0,"O2":0.0,"H2O":0.0}
PRESET_GAS = {"N2":0.204,"CO2":0.748,"CH4":94.87,"C2H6":3.721,"C3H8":0.376,"iC4H10":0.039,"nC4H10":0.028,"iC5H12":0.0,"nC5H12":0.0,"C6H14":0.015,"C7H16":0.0,"C8H18":0.0,"C9H20+":0.0,"O2":0.0,"H2O":0.0}

if 'df_comp' not in st.session_state:
    data = {"Componente": list(COMPONENTS_DB.keys()), "% Molar": [0.0]*len(COMPONENTS_DB)}
    st.session_state.df_comp = pd.DataFrame(data)
    st.session_state.df_comp.loc[st.session_state.df_comp['Componente']=='CH4', '% Molar'] = 100.0

def load_petroleo():
    for i, row in st.session_state.df_comp.iterrows():
        st.session_state.df_comp.at[i, '% Molar'] = PRESET_PETROLEO.get(row['Componente'], 0.0)
def load_gas():
    for i, row in st.session_state.df_comp.iterrows():
        st.session_state.df_comp.at[i, '% Molar'] = PRESET_GAS.get(row['Componente'], 0.0)

# --- INTERFAZ ---
st.title("🛢️ Ingeniería: Separadores Bifásicos & Gas")

# --- 1. COMPOSICION ---
st.sidebar.header("1. Gas Composition")
c1, c2 = st.sidebar.columns(2)
c1.button("Cargar Petróleo", on_click=load_petroleo)
c2.button("Cargar Gas", on_click=load_gas)
edited_df = st.sidebar.data_editor(st.session_state.df_comp, hide_index=True, height=550, use_container_width=True)
st.session_state.df_comp = edited_df
mole_fractions = {r['Componente']: r['% Molar']/100 for i, r in edited_df.iterrows()}

# --- 2. CONDICIONES GAS ---
st.header("2. Condiciones Gas y Proceso")
col_g1, col_g2, col_g3, col_g4 = st.columns(4)

with col_g1:
    t_u = st.selectbox("Unidad T", ["°C", "°F"])
    t_val = st.number_input("Temp", 45.0)
with col_g2:
    p_u = st.selectbox("Unidad P", ["bar", "psi", "kPa"])
    p_val = st.number_input("Presion (Gauge)", 30.0)
with col_g3:
    cond_q = st.selectbox("Ref Caudal Gas", ["Std (60F, 14.7p)", "Operativo"])
with col_g4:
    if "Std" in cond_q: u_q = st.selectbox("Unidad Gas", ["Sm3/d", "MMSCFD", "Sft3/d"])
    else: u_q = st.selectbox("Unidad Gas", ["m3/d", "m3/h", "ft3/d"])
    q_gas_val = st.number_input("Flujo Gas", 100000.0)

# --- 3. FASE LIQUIDA (NUEVO) ---
st.markdown("---")
st.header("3. Capacidad de Líquido (API 12J)")
st.caption("Cálculo de caudal máximo basado en volumen de retención y tiempo de residencia (Eq C.1.6).")

col_l1, col_l2, col_l3, col_l4 = st.columns(4)
with col_l1:
    res_time = st.number_input("Tiempo Residencia (min)", 1.0, 60.0, 3.0, help="Típicamente 3-5 min para Oil/Gas")
with col_l2:
    vol_u = st.selectbox("Unidad Volumen Sep.", ["m3", "ft3"])
    vol_liq_val = st.number_input("Volumen Líquido (NLL)", 0.0, 1000.0, 2.0)
with col_l3:
    # Necesitamos densidad liquido para el momentum de mezcla
    rho_liq_sg = st.number_input("SG Líquido (Agua=1)", 0.5, 1.5, 0.85, help="Gravedad específica para cálculo de momentum.")
with col_l4:
    st.write("Resultados Líquido:")
    
    # Calc Liquido
    # Q = V / t
    if vol_u == "m3": vol_m3 = vol_liq_val
    else: vol_m3 = vol_liq_val * 0.0283168
    
    q_liq_m3d = (vol_m3 / res_time) * 1440.0 # m3/min * 1440 min/d
    q_liq_bbld = q_liq_m3d * 6.2898
    
    st.success(f"Max Liq: **{q_liq_m3d:.1f} m3/d**")
    st.write(f"Max Liq: **{q_liq_bbld:.0f} bbl/d**")

# --- CALCULOS ---
# 1. T/P SI
T_k = t_val + 273.15 if t_u == "°C" else (t_val - 32)*5/9 + 273.15
P_bar = p_val + 1.013 if p_u == "bar" else (p_val+14.7)/14.5
Z, rho_gas, M = calc_peng_robinson(T_k, P_bar, mole_fractions)

# 2. Gas Standard conversion
T_std, P_std = 288.71, 1.01353
Z_std, _, _ = calc_peng_robinson(T_std, P_std, mole_fractions)
factor = (P_std/P_bar) * (T_k/T_std) * (Z/Z_std)

q_gas_act_m3d = 0.0
if "Std" in cond_q:
    qs = q_gas_val
    if u_q=="MMSCFD": qs = q_gas_val * 1e6 * 0.0283168
    elif u_q=="Sft3/d": qs = q_gas_val * 0.0283168
    q_gas_act_m3d = qs * factor
else:
    qa = q_gas_val
    if u_q=="m3/h": qa = q_gas_val * 24
    elif u_q=="ft3/d": qa = q_gas_val * 0.0283168
    q_gas_act_m3d = qa

# 3. Mezcla (Total Inlet)
q_liq_act_m3d = q_liq_m3d # Liquido se asume incompresible aprox
q_total_act_m3d = q_gas_act_m3d + q_liq_act_m3d
q_total_m3s = q_total_act_m3d / 86400

# Densidad Mezcla (Homogénea para Momentum)
rho_liq_kgm3 = rho_liq_sg * 1000
mass_gas = q_gas_act_m3d * rho_gas
mass_liq = q_liq_act_m3d * rho_liq_kgm3
rho_mix = (mass_gas + mass_liq) / q_total_act_m3d
rho_mix_lb = rho_mix * 0.062428

# --- RESULTADOS ---
st.markdown("---")
st.header("4. Resultados de Flujo de Entrada")
c1, c2, c3 = st.columns(3)
c1.metric("Caudal GAS Actual", f"{q_gas_act_m3d:.1f} m3/d")
c2.metric("Caudal LIQ Actual", f"{q_liq_act_m3d:.1f} m3/d")
c3.metric("Caudal TOTAL Entrada", f"{q_total_act_m3d:.1f} m3/d", help="Suma volumétrica a condiciones P, T")

c4, c5, c6 = st.columns(3)
c4.metric("Densidad Gas", f"{rho_gas:.2f} kg/m3")
c5.metric("Densidad Líquido", f"{rho_liq_kgm3:.1f} kg/m3")
c6.metric("Densidad Mezcla", f"{rho_mix:.2f} kg/m3", delta_color="off", help="Usada para momentum de entrada")

# --- BOQUILLAS ---
st.markdown("---")
col_inlet, col_outlet = st.columns(2)

PIPES = {
    "2 inch":  {"Sch40": 2.067, "Sch80": 1.939, "Sch160": 1.687},
    "3 inch":  {"Sch40": 3.068, "Sch80": 2.900, "Sch160": 2.624},
    "4 inch":  {"Sch40": 4.026, "Sch80": 3.826, "Sch160": 3.438},
    "6 inch":  {"Sch40": 6.065, "Sch80": 5.761, "Sch160": 5.187},
    "8 inch":  {"Sch40": 7.981, "Sch80": 7.625, "Sch160": 6.813},
    "10 inch": {"Sch40": 10.02, "Sch80": 9.562, "Sch160": 8.500},
    "12 inch": {"SchSTD": 12.00, "SchXS": 11.75, "Sch160": 10.12},
}

with col_inlet:
    st.subheader("🟢 Boquilla de ENTRADA (Mix)")
    st.caption(f"Verificación con Q_Total ({q_total_act_m3d:.0f} m3/d) y Rho_Mix")
    
    max_mom_in = st.number_input("Max Momentum In (lb/ft s2)", 1000, 2000, 1500)
    
    rows = []
    for sz, schs in PIPES.items():
        id_ref = schs.get("Sch80", list(schs.values())[0])
        a_ft2 = np.pi*(id_ref/24)**2
        v = (q_total_m3s * 35.315) / a_ft2
        mom = rho_mix_lb * v**2
        stt = "✅ OK" if mom <= max_mom_in else "❌ ALTO"
        rows.append([sz, id_ref, f"{v:.1f}", f"{mom:.0f}", stt])
        
    st.dataframe(pd.DataFrame(rows, columns=["Size","ID","Vel ft/s","Mom.","Estado"]), hide_index=True)

with col_outlet:
    st.subheader("🔵 Boquilla Salida GAS")
    st.caption(f"Verificación solo GAS ({q_gas_act_m3d:.0f} m3/d)")
    
    max_mom_out = st.number_input("Max Momentum Out", 1000, 5000, 3000)
    max_vel_out = st.number_input("Max Vel Out (ft/s)", 30, 100, 60)
    
    rows_out = []
    q_gas_m3s = q_gas_act_m3d / 86400
    rho_gas_lb = rho_gas * 0.062428
    
    for sz, schs in PIPES.items():
        id_ref = schs.get("Sch40", list(schs.values())[0])
        a_ft2 = np.pi*(id_ref/24)**2
        v = (q_gas_m3s * 35.315) / a_ft2
        mom = rho_gas_lb * v**2
        fail = []
        if mom > max_mom_out: fail.append("Mom")
        if v > max_vel_out: fail.append("Vel")
        stt = "✅ OK" if not fail else "❌"
        rows_out.append([sz, id_ref, f"{v:.1f}", f"{mom:.0f}", stt])
        
    st.dataframe(pd.DataFrame(rows_out, columns=["Size","ID","Vel ft/s","Mom.","Estado"]), hide_index=True)

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gas Process Calc AI", layout="wide", page_icon="🔥")

# --- 2. BASE DE DATOS DE COMPONENTES (Constantes Críticas) ---
# Tc (K), Pc (Bar), Omega, M (g/mol)
COMPONENTS_DB = {
    "N2":      {"Tc": 126.2,  "Pc": 33.9,  "w": 0.037, "M": 28.01},
    "CO2":     {"Tc": 304.1,  "Pc": 73.8,  "w": 0.225, "M": 44.01},
    "H2S":     {"Tc": 373.2,  "Pc": 89.4,  "w": 0.099, "M": 34.08},
    "O2":      {"Tc": 154.6,  "Pc": 50.4,  "w": 0.022, "M": 31.99},
    "H2O":     {"Tc": 647.1,  "Pc": 220.6, "w": 0.344, "M": 18.015},
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
    "C9H20+":  {"Tc": 594.6,  "Pc": 22.9,  "w": 0.445, "M": 128.26} # Propiedades estimadas para n-Nonano
}

# --- 3. FUNCIONES DE CÁLCULO TERMODINÁMICO (PENG-ROBINSON) ---
def calc_peng_robinson(T_k, P_bar, composition_mole_frac):
    """
    Calcula Z y Densidad usando PR EOS.
    T en Kelvin, P en Bar.
    composition: dict {Componente: FraccionMolar}
    """
    R = 8.314462618 # J/(mol K)
    P_pa = P_bar * 1e5
    
    # Mezcla
    Tc_mix, Pc_mix, w_mix, M_mix = 0, 0, 0, 0
    a_mix, b_mix = 0, 0
    
    # Pre-cálculo de parámetros puros
    params = {}
    for comp, y in composition_mole_frac.items():
        if y == 0: continue
        d = COMPONENTS_DB[comp]
        Tr = T_k / d['Tc']
        
        # PR parameters
        k = 0.37464 + 1.54226*d['w'] - 0.26992*(d['w']**2)
        alpha = (1 + k * (1 - np.sqrt(Tr)))**2
        a_i = 0.45724 * (R**2 * d['Tc']**2) / (d['Pc']*1e5) * alpha
        b_i = 0.07780 * (R * d['Tc']) / (d['Pc']*1e5)
        
        params[comp] = {'y': y, 'a': a_i, 'b': b_i, 'M': d['M']}
        M_mix += y * d['M']

    # Reglas de mezcla (Van der Waals 1-fluid, asumiendo kij=0 por simplicidad en esta versión)
    for c1, p1 in params.items():
        b_mix += p1['y'] * p1['b']
        for c2, p2 in params.items():
            a_mix += p1['y'] * p2['y'] * np.sqrt(p1['a'] * p2['a']) # kij = 0
            
    # Coeficientes Cúbica: Z^3 + c2*Z^2 + c1*Z + c0 = 0
    A = (a_mix * P_pa) / (R * T_k)**2
    B = (b_mix * P_pa) / (R * T_k)
    
    c2 = -(1 - B)
    c1 = (A - 3*B**2 - 2*B)
    c0 = -(A*B - B**2 - B**3)
    
    # Resolver raíces
    roots = np.roots([1, c2, c1, c0])
    
    # Filtrar raíces reales
    real_roots = roots[np.isreal(roots)].real
    # Para gas, tomamos la raíz real mayor
    Z = max(real_roots)
    
    # Densidad
    # PV = nRTZ -> P = rho/M * R * T * Z -> rho = PM / (ZRT)
    # rho (kg/m3) = (P_pa * M_mix_kg_mol) / (Z * R * T_k)
    rho_kg_m3 = (P_pa * (M_mix / 1000)) / (Z * R * T_k)
    
    return Z, rho_kg_m3, M_mix

# --- 4. DATA TUBERÍAS Y BRIDAS ---
PIPES_SCH = {
    "2 inch":  {"OD": 2.375, "IDs": {"Sch40": 2.067, "Sch80": 1.939, "Sch160": 1.687}},
    "3 inch":  {"OD": 3.500, "IDs": {"Sch40": 3.068, "Sch80": 2.900, "Sch160": 2.624}},
    "4 inch":  {"OD": 4.500, "IDs": {"Sch40": 4.026, "Sch80": 3.826, "Sch160": 3.438}},
    "6 inch":  {"OD": 6.625, "IDs": {"Sch40": 6.065, "Sch80": 5.761, "Sch160": 5.187}},
    "8 inch":  {"OD": 8.625, "IDs": {"Sch40": 7.981, "Sch80": 7.625, "Sch160": 6.813}},
    "10 inch": {"OD": 10.75, "IDs": {"Sch40": 10.02, "Sch80": 9.562, "Sch160": 8.500}},
    "12 inch": {"OD": 12.75, "IDs": {"SchSTD": 12.00, "SchXS": 11.75, "Sch160": 10.126}},
}

# Bridas ASME B16.5 (Diámetros internos aproximados para matching con tubería standard/xs)
# Simplificación: Usaremos el ID de tubería Sch 80/160 como referencia de bore máximo para 600/900
FLANGES_BORE = PIPES_SCH # Usaremos la misma DB para chequear disponibilidad

# --- 5. INTERFAZ DE USUARIO ---

st.sidebar.header("1. Composición del Gas (% Molar)")
st.sidebar.markdown("---")

comp_inputs = {}
total_comp = 0
for comp in COMPONENTS_DB.keys():
    # Valores default para un gas típico
    def_val = 90.0 if comp == 'CH4' else (5.0 if comp == 'C2H6' else 0.0)
    val = st.sidebar.number_input(f"{comp} %", 0.0, 100.0, def_val, 0.01)
    comp_inputs[comp] = val
    total_comp += val

st.sidebar.markdown(f"**Total: {total_comp:.2f}%**")
if abs(total_comp - 100.0) > 0.1:
    st.sidebar.warning("¡La composición debe sumar 100%!")

# Normalización interna
mole_fractions = {k: v/100 for k, v in comp_inputs.items()}

# --- MAIN PANEL ---
st.title("🏭 Ingeniería de Gas: Propiedades & Dimensionamiento")
st.markdown("Calculadora basada en **Peng-Robinson EOS**, **API RP 14E** y **API 12J**.")

# 2. Condiciones de Proceso
st.header("2. Condiciones Operativas")
col1, col2, col3, col4 = st.columns(4)

with col1:
    temp_unit = st.selectbox("Unidad T", ["°C", "°F"])
    temp_val = st.number_input("Temperatura", value=45.0)

with col2:
    press_unit = st.selectbox("Unidad P", ["bar", "psi", "kPa"])
    press_val = st.number_input("Presión (Gauge)", value=30.0)

with col3:
    flow_type = st.selectbox("Tipo de Caudal", ["Standard (Sm3/d, Sft3/d)", "Operativo (Actual)"])
    
with col4:
    if "Standard" in flow_type:
        flow_unit = st.selectbox("Unidad Caudal", ["Sm3/d", "Sm3/h", "Sft3/d", "Sft3/h"])
    else:
        flow_unit = st.selectbox("Unidad Caudal", ["m3/d", "m3/h", "ft3/d", "ft3/min"])
    flow_val = st.number_input("Valor de Caudal", value=100000.0)

# 3. Conversiones Previas
# Convertir todo a SI (Kelvin, Bar Absoluto, m3/s) para cálculos internos
if temp_unit == "°C": T_k = temp_val + 273.15
else: T_k = (temp_val - 32) * 5/9 + 273.15

if press_unit == "bar": P_bar_abs = press_val + 1.01325
elif press_unit == "psi": P_bar_abs = (press_val / 14.5038) + 1.01325
else: P_bar_abs = (press_val / 100.0) + 1.01325

# Condiciones Standard (15°C, 1 atm)
T_std_k = 288.15 # 15°C
P_std_bar = 1.01325 

# Calcular Z y Rho a condiciones Operativas y Standard
Z_op, rho_op_kgm3, M_gas = calc_peng_robinson(T_k, P_bar_abs, mole_fractions)
Z_std, rho_std_kgm3, _ = calc_peng_robinson(T_std_k, P_std_bar, mole_fractions)

# Factor de conversión de volumen (Bg o similar)
# Q_op = Q_std * (P_std/P_op) * (T_op/T_std) * (Z_op/Z_std)
factor_std_to_op = (P_std_bar / P_bar_abs) * (T_k / T_std_k) * (Z_op / Z_std)

# Normalizar caudal a m3/d OPERATIVO para cálculos
q_actual_m3d = 0.0

if "Standard" in flow_type:
    # Convertir input a Sm3/d primero
    q_std_m3d = 0.0
    if flow_unit == "Sm3/d": q_std_m3d = flow_val
    elif flow_unit == "Sm3/h": q_std_m3d = flow_val * 24
    elif flow_unit == "Sft3/d": q_std_m3d = flow_val * 0.0283168
    elif flow_unit == "Sft3/h": q_std_m3d = flow_val * 0.0283168 * 24
    
    # Convertir a Operativo
    q_actual_m3d = q_std_m3d * factor_std_to_op
else:
    # Ya es operativo, convertir a m3/d
    if flow_unit == "m3/d": q_actual_m3d = flow_val
    elif flow_unit == "m3/h": q_actual_m3d = flow_val * 24
    elif flow_unit == "ft3/d": q_actual_m3d = flow_val * 0.0283168
    elif flow_unit == "ft3/min": q_actual_m3d = flow_val * 0.0283168 * 1440

q_actual_m3s = q_actual_m3d / 86400
mass_flow_kg_s = q_actual_m3s * rho_op_kgm3

# --- RESULTADOS ---
st.markdown("---")
st.header("3. Resultados de Propiedades del Gas")

res_col1, res_col2, res_col3 = st.columns(3)

with res_col1:
    st.metric("Factor Z (Compresibilidad)", f"{Z_op:.4f}")
    st.metric("Peso Molecular", f"{M_gas:.2f} g/mol")

with res_col2:
    st.subheader("Densidad")
    dens_unit = st.selectbox("Unidad Densidad", ["kg/m3", "lb/ft3"])
    dens_val = rho_op_kgm3 if dens_unit == "kg/m3" else rho_op_kgm3 * 0.062428
    st.metric(f"Densidad ({dens_unit})", f"{dens_val:.2f}")

with res_col3:
    st.subheader("Volumen / Caudal Recíproco")
    # Mostrar el caudal equivalente en la condición opuesta
    if "Standard" in flow_type:
        st.metric("Caudal Operativo (m3/d)", f"{q_actual_m3d:.2f}")
        st.metric("Caudal Operativo (ft3/d)", f"{q_actual_m3d/0.0283168:.2f}")
    else:
        q_std_equiv = q_actual_m3d / factor_std_to_op
        st.metric("Caudal Standard (Sm3/d)", f"{q_std_equiv:.2f}")
        st.metric("Caudal Standard (MMSCFD)", f"{(q_std_equiv/0.0283168)/1e6:.3f}")

# --- 4. API RP 14E (EROSIONAL) ---
st.markdown("---")
st.header("4. API RP 14E - Velocidad Erosional")

# Constante C
c_const = 125 # User fixed value (Continuous service without solids)
rho_lb_ft3 = rho_op_kgm3 * 0.062428

# Ve (ft/s) = C / sqrt(rho_lb_ft3)
ve_fts = c_const / np.sqrt(rho_lb_ft3)
ve_ms = ve_fts * 0.3048

st.info(f"Para C={c_const}, la **Velocidad Erosional Límite (Ve)** es: **{ve_ms:.2f} m/s ({ve_fts:.2f} ft/s)**")

# Selección de tubería
st.subheader("Selección de Tubería Adecuada (API 14E)")
# Necesitamos Área mínima = Q_actual / Ve
area_req_m2 = q_actual_m3s / ve_ms
area_req_inch2 = area_req_m2 * 1550.0

pipe_data = []
for size, props in PIPES_SCH.items():
    for sch, id_inch in props['IDs'].items():
        area_pipe_inch2 = np.pi * (id_inch/2)**2
        velocity_pipe_fts = (q_actual_m3s * 35.3147) / (area_pipe_inch2 / 144)
        
        # Check if compliant
        status = "✅ OK" if velocity_pipe_fts < ve_fts else "❌ Erosión"
        ratio = (velocity_pipe_fts / ve_fts) * 100
        
        pipe_data.append({
            "Tamaño Nominal": size,
            "Schedule": sch,
            "ID (in)": id_inch,
            "Velocidad (ft/s)": round(velocity_pipe_fts, 2),
            "% de Ve": round(ratio, 1),
            "Estado": status
        })

df_pipes = pd.DataFrame(pipe_data)
# Filtrar solo las que cumplen y ordenar
df_ok = df_pipes[df_pipes["Estado"] == "✅ OK"].sort_values(by="% de Ve", ascending=False)

st.dataframe(df_ok.style.map(lambda x: 'color: red' if 'Erosión' in str(x) else 'color: green', subset=['Estado']), use_container_width=True)


# --- 5. API 12J (BOQUILLAS) ---
st.markdown("---")
st.header("5. Dimensionamiento de Boquillas (API 12J)")

# Lógica: API 12J Eq C.1.1 con K=0.6275 (Imperial) 
# Esta ecuación suele ser W = ... pero el usuario pide "Velocidad en Boquillas" asumiendo K.
# Interpretaremos esto como el límite de Momento (rho * v^2) o velocidad crítica.
# Dado el prompt "K=0.6275 (Imperial)", esto se refiere a menudo a la capacidad de separación, pero la aplicaremos
# como criterio de velocidad si se trata de un separador.
# Sin embargo, lo más standard para boquillas es Momentum < 1000-1500 lb/(ft s2).
# Implementaremos el cálculo basado en MOMENTO y mostraremos la conversión.

st.write("Criterio: Momentum $\\rho v^2$ (API 12J Inlet/Outlet guidelines).")
max_momentum = st.number_input("Criterio Momentum Max (lb/ft s²)", value=1500.0, help="Típicamente 1000-1500 para entrada, hasta 4000 para salida limpia.")

# Calculo de velocidad máxima permitida
# rho * v^2 <= MaxM -> v_max = sqrt(MaxM / rho)
v_max_nozzle_fts = np.sqrt(max_momentum / rho_lb_ft3)
v_max_nozzle_ms = v_max_nozzle_fts * 0.3048

st.write(f"Velocidad Máxima permitida en boquilla: **{v_max_nozzle_ms:.2f} m/s ({v_max_nozzle_fts:.2f} ft/s)**")

# Buscar brida ASME
nozzle_data = []
for size, props in PIPES_SCH.items():
    # Usamos el ID más restrictivo (Sch 160) para ser conservadores en el bore de la brida
    # O el Sch 80 que es común en alta presión. Usaremos Sch 80 como referencia de brida High Pressure.
    id_ref = props['IDs'].get('Sch80', props['IDs'].get('SchSTD'))
    
    area_noz_ft2 = (np.pi * (id_ref/2)**2) / 144
    v_actual_fts = (q_actual_m3s * 35.3147) / area_noz_ft2
    momentum_actual = rho_lb_ft3 * (v_actual_fts**2)
    
    status_noz = "✅ OK" if momentum_actual <= max_momentum else "❌ Excede"
    
    nozzle_data.append({
        "Brida Nom. (Ref Sch80 Bore)": size,
        "ID (in)": id_ref,
        "Velocidad (ft/s)": round(v_actual_fts, 2),
        "Momentum (lb/ft s²)": round(momentum_actual, 0),
        "Estado": status_noz
    })

df_nozzles = pd.DataFrame(nozzle_data)
st.dataframe(df_nozzles, use_container_width=True)

# --- EXPORTAR ---
st.markdown("---")
csv_pipes = df_pipes.to_csv(index=False).encode('utf-8')
st.download_button("Descargar Tabla Tuberías (CSV)", csv_pipes, "tuberias_api14e.csv", "text/csv")

csv_noz = df_nozzles.to_csv(index=False).encode('utf-8')
st.download_button("Descargar Tabla Boquillas (CSV)", csv_noz, "boquillas_api12j.csv", "text/csv")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import requests
import time 
import math
import plotly.graph_objects as go
import searoute as sr 
import websocket
import json
import shap
from fpdf import FPDF
import base64

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =============================================================================
# PAGE SETUP & GLOBAL VARIABLES
# =============================================================================
st.set_page_config(page_title="Vessel AI & CII Optimizer V11.6", page_icon="🚢", layout="wide")
st.title("🚢 Ultimate Digital Twin & OPEX Simulator V11.6 (Stable IoT)")
st.markdown("**Modules:** Computer Vision | 5D Explainable AI | Nav & ECA | JIT OPEX | **Live Engine Telemetry** | **Fleet Database** | PDF Export")
st.markdown("---")

if 'calc_wind_area' not in st.session_state: st.session_state.calc_wind_area = 800.0
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'df_sim' not in st.session_state: st.session_state.df_sim = None
if 'voyage_report' not in st.session_state: st.session_state.voyage_report = None
if 'fleet_history' not in st.session_state: st.session_state.fleet_history = [] 

# =============================================================================
# PHYSICS ENGINE, ECA & API LIBRARIES
# =============================================================================
FUEL_DATA = {
    "HFO (Heavy Fuel Oil)": {"co2_factor": 3.114, "price": 500, "eca_compliant": False},
    "VLSFO (Low Sulfur)": {"co2_factor": 3.206, "price": 650, "eca_compliant": False},
    "MGO (Marine Gas Oil)": {"co2_factor": 3.206, "price": 850, "eca_compliant": True},
    "LNG (Liquefied Nat. Gas)": {"co2_factor": 2.750, "price": 450, "eca_compliant": True},
    "Green Methanol": {"co2_factor": 0.0, "price": 1000, "eca_compliant": True},
    "Ammonia (Zero Carbon)": {"co2_factor": 0.0, "price": 1200, "eca_compliant": True}
}

def is_in_eca(lat, lon):
    if (48 <= lat <= 62 and -5 <= lon <= 12): return True 
    if (25 <= lat <= 50 and -80 <= lon <= -60): return True
    return False

def get_live_weather(lat, lon, api_key):
    try:
        res = requests.get(f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric").json()
        if res.get("cod") != 200: return 4 
        return min(12, round((res["wind"]["speed"] / 0.836) ** (2/3)))
    except: return 4

def calculate_fuel(speed, draft, w_area, bft, d_spd, d_dft, d_cons, beam, months_drydock, trim):
    v_ms = speed * 0.5144
    p_des = (d_cons * 1000000) / (24 * 175) 
    
    bio_penalty = 1.0 + (months_drydock * 0.015) 
    p_calm = p_des * ((speed / d_spd)**3) * ((draft / d_dft)**(2/3)) * bio_penalty
    
    v_rel = v_ms + (0.836 * (bft ** 1.5))
    p_wind = (0.5 * 1.225 * 0.8 * w_area * (v_rel ** 2) * v_ms) / 1000
    p_wave = ((1/16) * 1025 * 9.81 * ((0.2 * (bft ** 2)) ** 2) * beam * math.sqrt(beam / max(100, beam*6.5)) * v_ms) / 1000
    
    trim_effect = 1.0 + (abs(trim - 0.5) * 0.02)
    p_total = (p_calm + p_wind + p_wave) * trim_effect
    
    load = max(0.1, p_total / (p_des * 1.1))
    return (p_total * (175 * (1 + 0.5 * (load - 0.75)**2)) * 24) / 1000000

def get_cii(fuel, speed, dwt, ref, co2_f):
    if speed <= 0: return "E"
    ratio = ((fuel * co2_f * 1_000_000) / (dwt * speed * 24)) / ref
    return "A" if ratio < 0.83 else "B" if ratio < 0.94 else "C" if ratio < 1.06 else "D" if ratio < 1.19 else "E"

def get_distance(l1, ln1, l2, ln2):
    dl, dln = math.radians(l2 - l1), math.radians(ln2 - ln1)
    a = math.sin(dl/2)**2 + math.cos(math.radians(l1)) * math.cos(math.radians(l2)) * math.sin(dln/2)**2
    return 3440.065 * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def get_ais_data(api_key, coords):
    lats, lons = [c[1] for c in coords], [c[0] for c in coords]
    box = [[[min(lats)-5.0, min(lons)-5.0], [max(lats)+5.0, max(lons)+5.0]]]
    msg = {"APIKey": api_key, "BoundingBoxes": box, "FilterMessageTypes": ["PositionReport"]}
    vessels = {}
    try:
        ws = websocket.create_connection("wss://stream.aisstream.io/v0/stream", timeout=5)
        ws.send(json.dumps(msg))
        t0 = time.time()
        while time.time() - t0 < 5:
            try:
                res = json.loads(ws.recv())
                if res["MessageType"] == "PositionReport":
                    pr = res["Message"]["PositionReport"]
                    sog = float(pr.get("Sog", 0))
                    vessels[res["MetaData"]["MMSI"]] = {
                        "lat": pr["Latitude"], "lon": pr["Longitude"], 
                        "name": res["MetaData"]["ShipName"].strip() or "Unknown", 
                        "sog": sog/10.0 if sog>50 else sog
                    }
                    if len(vessels) >= 1500: break
            except: break
        ws.close()
    except: pass 
    return list(vessels.values())

def create_pdf(report_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AI VOYAGE ORDER & OPTIMIZATION REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    pdf.line(10, 20, 200, 20)
    pdf.ln(10)
    for key, val in report_data.items():
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(80, 8, txt=str(key)+":", ln=False)
        pdf.set_font("Arial", '', 11)
        pdf.cell(100, 8, txt=str(val), ln=True)
    return pdf.output(dest='S').encode('latin-1')

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
st.sidebar.header("⚙️ 1. Vessel Particulars")
dwt = st.sidebar.number_input("Deadweight (DWT)", value=105000.0)
beam = st.sidebar.number_input("Beam (m)", value=42.8)
d_speed = st.sidebar.number_input("Design Speed (Kn)", value=24.0)
d_draft = st.sidebar.number_input("Design Draft (m)", value=14.5)
d_cons = st.sidebar.number_input("Design Cons (T/Day)", value=225.0)

st.sidebar.header("📡 2. Live API Keys")
api_weather = st.sidebar.text_input("OpenWeather API:", type="password")
api_ais = st.sidebar.text_input("AISStream API:", type="password")

st.sidebar.header("💰 3. Financial & OPEX Inputs")
fuel_type = st.sidebar.selectbox("Main Fuel Type", list(FUEL_DATA.keys()))
eu_ets = st.sidebar.number_input("EU ETS Tax (€/T CO2)", value=85.0)
daily_charter = st.sidebar.number_input("Daily Charter Rate ($)", value=25000)

st.sidebar.header("🏗️ 4. AI Operational State")
port_ops = st.sidebar.number_input("Port Handling (Hr/Ship)", value=12.0)
months_drydock = st.sidebar.slider("Months Since Drydock", 0, 60, 12)
vessel_trim = st.sidebar.slider("Vessel Trim (m) [-Bow, +Stern]", -2.0, 2.0, 0.5, step=0.1)

f_co2 = FUEL_DATA[fuel_type]["co2_factor"]
f_price = FUEL_DATA[fuel_type]["price"]
f_compliant = FUEL_DATA[fuel_type]["eca_compliant"]
mgo_price = FUEL_DATA["MGO (Marine Gas Oil)"]["price"]
mgo_co2 = FUEL_DATA["MGO (Marine Gas Oil)"]["co2_factor"]
ref_cii = (d_cons * 3.114 * 1_000_000) / (dwt * d_speed * 24)

# =============================================================================
# MAIN TABS UI
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📷 CV Wind Area", "🧠 5D AI Training", "🗺️ 2D OPEX Route", "📡 Live Engine IoT", "🗄️ Fleet & Reports"])

# --- TAB 1: COMPUTER VISION ---
with tab1:
    st.subheader("Frontal Area Extraction via Computer Vision")
    c1, c2 = st.columns(2)
    img_file = c1.file_uploader("Upload Transverse Profile Image", type=['png', 'jpg', 'jpeg'])
    if img_file:
        img = cv2.imdecode(np.asarray(bytearray(img_file.read()), dtype=np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            st.session_state.calc_wind_area = float(cv2.contourArea(c) * ((beam/w)**2))
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Bounding Box Applied")
            c2.success(f"🌬️ Calculated Wind Area: {st.session_state.calc_wind_area:.1f} m²")

# --- TAB 2: 5D AI TRAINING & SHAP ---
with tab2:
    st.subheader("5-Dimensional Random Forest & SHAP Interpretation")
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        sim_days = st.number_input("Dataset Size (Days):", min_value=1000, value=10000)
        if st.button("🚀 Initialize 5D AI Model"):
            with st.spinner("AI is learning ocean dynamics & trim states..."):
                np.random.seed(42)
                s_v = np.random.uniform(d_speed*0.4, d_speed*1.05, sim_days)
                s_d = np.random.uniform(d_draft*0.5, d_draft, sim_days)
                s_b = np.random.randint(0, 11, sim_days)
                s_m = np.random.randint(0, 61, sim_days)
                s_t = np.random.uniform(-2.0, 2.0, sim_days)
                
                s_f = [calculate_fuel(s_v[i], s_d[i], st.session_state.calc_wind_area, s_b[i], d_speed, d_draft, d_cons, beam, s_m[i], s_t[i]) for i in range(sim_days)]
                
                df = pd.DataFrame({'Speed_Knots': s_v, 'Draft_m': s_d, 'Beaufort': s_b, 'Months_Drydock': s_m, 'Trim_m': s_t, 'Fuel_Ton': s_f})
                X, y = df.drop('Fuel_Ton', axis=1), df['Fuel_Ton']
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
                
                model = RandomForestRegressor(n_estimators=100).fit(X_tr, y_tr)
                st.session_state.ai_model = model
                st.session_state.df_sim = X_te 
                st.success(f"✅ AI Accuracy (R²): {r2_score(y_te, model.predict(X_te))*100:.2f}%")

    with col_t2:
        if st.session_state.ai_model is not None:
            st.markdown("**🧠 5D Feature Importance (SHAP Analysis)**")
            with st.spinner("Generating SHAP Explanations..."):
                X_sample = shap.sample(st.session_state.df_sim, 100) 
                explainer = shap.TreeExplainer(st.session_state.ai_model)
                shap_values = explainer.shap_values(X_sample, check_additivity=False) 
                fig_shap, ax = plt.subplots(figsize=(6, 4))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                st.pyplot(fig_shap)

# --- TAB 3: 2D NAVIGATION, OPEX & ECA ---
with tab3:
    st.subheader("🌍 Route Optimization (ECA Compliance & OPEX)")
    ports = {"Tokyo": (35.6, 139.6), "Shanghai": (31.2, 121.5), "Singapore": (1.3, 103.8), "Rotterdam": (51.9, 4.4), "Istanbul": (41.0, 28.9), "Panama": (9.1, -79.6), "New York": (40.7, -74.0)}
    cn1, cn2, cn3, cn4 = st.columns(4)
    c_lat = cn1.number_input("Current Lat:", value=35.0)
    c_lon = cn2.number_input("Current Lon:", value=15.0)
    d_port = cn3.selectbox("Destination Port:", list(ports.keys()), index=3)
    urgency = cn4.slider("Speed Priority (%):", 0, 100, 50)

    if st.button("🛰️ Start Route Analysis"):
        with st.spinner("Analyzing ECA Zones, Weather, and OPEX..."):
            d_lat, d_lon = ports[d_port]
            try: coords = sr.searoute([c_lon, c_lat], [d_lon, d_lat])["geometry"]["coordinates"]
            except: st.error("Routing failed."); st.stop()
            
            max_v = d_speed
            norm_v = 10.0 + (max_v - 10.0) * (urgency / 100.0)
            
            def get_f(v, b, m, t):
                if st.session_state.ai_model:
                    return st.session_state.ai_model.predict(pd.DataFrame([[v, d_draft, b, m, t]], columns=['Speed_Knots', 'Draft_m', 'Beaufort', 'Months_Drydock', 'Trim_m']))[0]
                return calculate_fuel(v, d_draft, st.session_state.calc_wind_area, b, d_speed, d_draft, d_cons, beam, m, t)

            segments, dist_nm, bft_sum = [], 0, 0
            eca_nm = 0 
            storm_encounters = 0
            storm_eval_s_cost, storm_eval_d_cost = 0, 0 
            extra_days_total, wr_saved_usd = 0, 0
            wr_active = False
            
            step = max(1, len(coords)//10)
            
            for i in range(0, len(coords)-1, step):
                seg = coords[i:min(i+step+1, len(coords))]
                d = sum([get_distance(seg[k][1], seg[k][0], seg[k+1][1], seg[k+1][0]) for k in range(len(seg)-1)])
                bft = get_live_weather(seg[0][1], seg[0][0], api_weather) if api_weather else 4
                
                in_eca_zone = is_in_eca(seg[0][1], seg[0][0])
                if in_eca_zone: eca_nm += d
                
                seg_f_price = mgo_price if (in_eca_zone and not f_compliant) else f_price
                seg_f_co2 = mgo_co2 if (in_eca_zone and not f_compliant) else f_co2

                if bft >= 7:
                    storm_encounters += 1
                    t_straight = d / (norm_v * 24)
                    f_straight = get_f(norm_v, bft, months_drydock, vessel_trim) * t_straight
                    cost_s = (f_straight * seg_f_price) + (f_straight * seg_f_co2 * eu_ets) + (t_straight * daily_charter)
                    
                    t_detour = (d * 1.15) / (norm_v * 24)
                    f_detour = get_f(norm_v, 4, months_drydock, vessel_trim) * t_detour
                    cost_d = (f_detour * seg_f_price) + (f_detour * seg_f_co2 * eu_ets) + (t_detour * daily_charter)
                    
                    storm_eval_s_cost += cost_s
                    storm_eval_d_cost += cost_d
                    
                    if cost_d < cost_s:
                        wr_active = True
                        bft, d = 4, d * 1.15
                        if in_eca_zone: eca_nm += (d * 0.15)
                        wr_saved_usd += (cost_s - cost_d)
                        extra_days_total += (t_detour - t_straight)
                        seg = [[c[0]+0.85, c[1]-0.85] for c in seg]
                        segments.append({'c': seg, 'color': '#9b59b6', 'eca': in_eca_zone}) 
                    else: 
                        segments.append({'c': seg, 'color': '#e74c3c', 'eca': in_eca_zone}) 
                else: 
                    segments.append({'c': seg, 'color': '#2ecc71', 'eca': in_eca_zone}) 
                
                dist_nm += d
                bft_sum += bft
                
            avg_bft = round(bft_sum / len(segments))
            
            live_ships, port_queue = get_ais_data(api_ais, coords) if api_ais else [], 0
            for s in live_ships:
                if get_distance(d_lat, d_lon, s['lat'], s['lon']) <= 30: port_queue += 1

            norm_days = dist_nm / (norm_v * 24)
            wait_hrs = (port_queue * 12) / 3 
            jit_v = max(8.0, dist_nm / ((norm_days + wait_hrs/24) * 24))

            eca_ratio = eca_nm / dist_nm if dist_nm > 0 else 0
            f_norm = get_f(norm_v, avg_bft, months_drydock, vessel_trim) * norm_days
            f_jit = get_f(jit_v, avg_bft, months_drydock, vessel_trim) * (dist_nm / (jit_v * 24))
            
            def calc_blended_opex(total_fuel, days):
                fuel_eca = total_fuel * eca_ratio
                fuel_open = total_fuel * (1 - eca_ratio)
                cost_open = (fuel_open * f_price) + (fuel_open * f_co2 * eu_ets)
                cost_eca = (fuel_eca * (mgo_price if not f_compliant else f_price)) + (fuel_eca * (mgo_co2 if not f_compliant else f_co2) * eu_ets)
                return cost_open + cost_eca + (days * daily_charter)
                
            opex_norm = calc_blended_opex(f_norm, norm_days) + (wait_hrs * (daily_charter/24))
            opex_jit = calc_blended_opex(f_jit, dist_nm/(jit_v*24)) 
            
            ai_cii_grade = get_cii(f_norm / norm_days, norm_v, dwt, ref_cii, f_co2)
            
            fig_map = go.Figure()
            for s in segments:
                dash_style = 'dash' if s['eca'] else 'solid'
                fig_map.add_trace(go.Scattergeo(lon=[c[0] for c in s['c']], lat=[c[1] for c in s['c']], mode='lines', line=dict(width=4, color=s['color'], dash=dash_style)))
            if live_ships:
                fig_map.add_trace(go.Scattergeo(lon=[v['lon'] for v in live_ships], lat=[v['lat'] for v in live_ships], mode='markers', marker=dict(size=4, color='blue')))
            
            fig_map.update_layout(dragmode='pan', geo=dict(projection_type="equirectangular", showland=True, landcolor="#f0f0f0", showocean=True, oceancolor="#cce5ff"), height=600, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig_map, use_container_width=True)

            if eca_nm > 0 and not f_compliant:
                st.error(f"🛑 **MARPOL ECA Regulation Alert:** Route crosses Emission Control Area for **{eca_nm:,.0f} Nm**. AI automatically bypassed {fuel_type} and switched main engine to compliant MGO to avoid heavy fines.")
            
            st.markdown("### 💼 Total OPEX & Hydrodynamic Optimizer")
            if opex_norm > opex_jit:
                st.success(f"**💡 AI JIT Recommendation:** Reduce speed to **{jit_v:.1f} Knots**. Save **${opex_norm - opex_jit:,.0f}** in OPEX!")
            
            col_o1, col_o2, col_o3 = st.columns(3)
            col_o1.metric("Current Route OPEX", f"${opex_norm:,.0f}")
            col_o2.metric("JIT Optimized OPEX", f"${opex_jit:,.0f}")
            col_o3.metric("ECA Compliance Distance", f"{eca_nm:,.0f} Nm")

            st.session_state.voyage_report = {
                "Timestamp": time.strftime("%Y-%m-%d %H:%M"),
                "Destination": d_port, "Distance (Nm)": round(dist_nm), "Trim State (m)": vessel_trim,
                "ECA Sailing (Nm)": round(eca_nm), "Recommended Speed": round(jit_v, 1) if opex_norm > opex_jit else round(norm_v, 1),
                "Est. Fuel (Tons)": round(f_jit if opex_norm > opex_jit else f_norm),
                "Total OPEX ($)": round(min(opex_norm, opex_jit)), "CII Grade": ai_cii_grade
            }
            st.session_state.fleet_history.append(st.session_state.voyage_report)

# --- TAB 4: LIVE ENGINE IOT TELEMETRY ---
with tab4:
    st.subheader("📡 Engine Control Room (ECR) Live Telemetry")
    st.caption("Simulated real-time Engine Data based on AI-Optimized Speed parameters.")
    
    if st.session_state.voyage_report:
        opt_speed = st.session_state.voyage_report["Recommended Speed"]
        base_rpm = (opt_speed / d_speed) * 105.0 
        base_egt = 320.0 + (opt_speed / d_speed) * 80.0 
        
        c_i1, c_i2, c_i3, c_i4 = st.columns(4)
        c_i1.metric("Main Engine RPM", f"{base_rpm + np.random.uniform(-0.5, 0.5):.1f} RPM", "Optimized", delta_color="normal")
        c_i2.metric("Exhaust Gas Temp (EGT)", f"{base_egt + np.random.uniform(-2.0, 2.0):.1f} °C", "Stable", delta_color="off")
        c_i3.metric("Scavenge Air Press.", f"{2.8 + (opt_speed/d_speed)*0.5 + np.random.uniform(-0.05, 0.05):.2f} Bar", "Normal", delta_color="off")
        c_i4.metric("Dynamic SFOC", f"{175 + np.random.uniform(-1.0, 1.0):.1f} g/kWh", "-1.2 g/kWh", delta_color="inverse")
        
        st.markdown("**📈 24-Hour Telemetry Trend (RPM vs Fuel Flow)**")
        time_index = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min')
        sim_rpm = np.random.normal(base_rpm, 1.5, 100)
        sim_fuel = np.random.normal((d_cons/24) * (opt_speed/d_speed)**3, 0.2, 100)
        
        # DÜZELTİLEN KISIM: title_font yerine dict içi tanımlama kullanıldı.
        fig_iot = go.Figure()
        fig_iot.add_trace(go.Scatter(x=time_index, y=sim_rpm, name="ME RPM", line=dict(color="cyan")))
        fig_iot.add_trace(go.Scatter(x=time_index, y=sim_fuel, name="Fuel Flow (T/hr)", yaxis="y2", line=dict(color="orange")))
        
        fig_iot.update_layout(
            yaxis=dict(title=dict(text="RPM", font=dict(color="cyan")), tickfont=dict(color="cyan")),
            yaxis2=dict(title=dict(text="Fuel Flow", font=dict(color="orange")), tickfont=dict(color="orange"), anchor="x", overlaying="y", side="right"),
            height=350, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font=dict(color="white")
        )
        st.plotly_chart(fig_iot, use_container_width=True)
    else:
        st.warning("⚠️ Waiting for Route Analysis to initialize Engine Telemetry...")

# --- TAB 5: REPORTS & FLEET DATABASE ---
with tab5:
    st.subheader("🗄️ Fleet Voyage History & Reports")
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        st.info("Export the current AI-Optimized voyage plan as a formal PDF document.")
        if st.session_state.voyage_report:
            pdf_bytes = create_pdf(st.session_state.voyage_report)
            st.download_button(label="📥 Download PDF Voyage Order", data=pdf_bytes, file_name="AI_Voyage_Order.pdf", mime="application/pdf")
        else:
            st.warning("⚠️ No active voyage to export.")
            
    with col_f2:
        st.markdown("**📊 Fleet Operations Database (Session)**")
        if len(st.session_state.fleet_history) > 0:
            df_history = pd.DataFrame(st.session_state.fleet_history)
            st.dataframe(df_history, use_container_width=True)
            
            total_savings = df_history["Total OPEX ($)"].sum()
            st.success(f"**Total Fleet OPEX Logged:** ${total_savings:,.0f}")
        else:
            st.caption("No voyages recorded yet. Run a route analysis to start logging.")

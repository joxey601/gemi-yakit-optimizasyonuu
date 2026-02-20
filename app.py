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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Gemi Yakıt & CII Optimizasyonu V9.1",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Gemi Performans ve Yakıt Simülatörü V9.1 (Final Stability)")
st.markdown("**Modüller:** Computer Vision | Big Data | Machine Learning | Live AIS | JIT Logistics | Predictive Maintenance | EU ETS | Weather Routing")
st.markdown("---")

# Session State Başlatmaları
if 'calc_wind_area' not in st.session_state:
    st.session_state.calc_wind_area = 800.0
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None

# =============================================================================
# BÖLÜM 1: AKADEMİK FİZİK MOTORU & DİJİTAL İKİZ MODÜLLERİ
# =============================================================================

FUEL_DATA = {
    "HFO (Ağır Yakıt)": {"co2_factor": 3.114, "price_per_ton": 500},
    "VLSFO (Düşük Sülfür)": {"co2_factor": 3.206, "price_per_ton": 650},
    "MGO (Dizel)": {"co2_factor": 3.206, "price_per_ton": 850},
    "LNG (Sıvı Doğalgaz)": {"co2_factor": 2.750, "price_per_ton": 450},
    "Green Methanol": {"co2_factor": 0.0, "price_per_ton": 1000},
    "Ammonia (Amonyak)": {"co2_factor": 0.0, "price_per_ton": 1200}
}

def get_live_weather_by_coords(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url).json()
        if response.get("cod") != 200: return None, 4 
        wind_speed_ms = response["wind"]["speed"]
        bft = round((wind_speed_ms / 0.836) ** (2/3))
        if bft > 12: bft = 12 
        return wind_speed_ms, bft
    except: return None, 4

def calculate_instant_fuel(speed, draft, wind_area, beaufort, des_spd, des_dft, des_cons, beam, months_since_drydock):
    v_ship_ms = speed * 0.5144
    p_des_kw = (des_cons * 1000000) / (24 * 175) 
    biofouling_penalty = 1.0 + (months_since_drydock * 0.015) 
    p_calm_kw = p_des_kw * ((speed / des_spd)**3) * ((draft / des_dft)**(2/3)) * biofouling_penalty
    wind_speed_ms = 0.836 * (beaufort ** 1.5)
    v_rel_ms = v_ship_ms + wind_speed_ms 
    rho_air = 1.225 
    c_aa = 0.8 
    r_aa_newton = 0.5 * rho_air * c_aa * wind_area * (v_rel_ms ** 2)
    p_wind_kw = (r_aa_newton * v_ship_ms) / 1000
    rho_water = 1025 
    g = 9.81
    h_s = 0.2 * (beaufort ** 2) 
    l_wl = beam * 6.5 
    if l_wl <= 0: l_wl = 100
    r_wave_newton = (1/16) * rho_water * g * (h_s ** 2) * beam * math.sqrt(beam / l_wl)
    p_wave_kw = (r_wave_newton * v_ship_ms) / 1000
    p_total_kw = p_calm_kw + p_wind_kw + p_wave_kw
    mcr_kw = p_des_kw * 1.1 
    load = p_total_kw / mcr_kw
    if load < 0.1: load = 0.1
    sfoc_dyn = 175 * (1 + 0.5 * (load - 0.75)**2)
    daily_fuel_ton = (p_total_kw * sfoc_dyn * 24) / 1000000
    return daily_fuel_ton

def calculate_cii_grade(fuel_ton, speed_knots, dwt, ref_cii, fuel_co2_factor):
    if speed_knots <= 0: return 9999, "E"
    attained_cii = (fuel_ton * fuel_co2_factor * 1_000_000) / (dwt * speed_knots * 24)
    if ref_cii <= 0: return attained_cii, "A"
    ratio = attained_cii / ref_cii
    if ratio < 0.83: return attained_cii, "A"
    elif ratio < 0.94: return attained_cii, "B"
    elif ratio < 1.06: return attained_cii, "C"
    elif ratio < 1.19: return attained_cii, "D"
    else: return attained_cii, "E"

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_ais_snapshot_route(api_key, route_coords, margin_deg=3.0, timeout=5, max_vessels=1500):
    lats = [c[1] for c in route_coords]
    lons = [c[0] for c in route_coords]
    min_lat, max_lat = min(lats) - margin_deg, max(lats) + margin_deg
    min_lon, max_lon = min(lons) - margin_deg, max(lons) + margin_deg
    bounding_box = [[[min_lat, min_lon], [max_lat, max_lon]]]
    subscribe_message = {"APIKey": api_key, "BoundingBoxes": bounding_box, "FilterMessageTypes": ["PositionReport"]}
    vessels = {}
    try:
        ws = websocket.create_connection("wss://stream.aisstream.io/v0/stream", timeout=timeout)
        ws.send(json.dumps(subscribe_message))
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = ws.recv()
                msg = json.loads(result)
                if msg["MessageType"] == "PositionReport":
                    mmsi = msg["MetaData"]["MMSI"]
                    vlat = msg["Message"]["PositionReport"]["Latitude"]
                    vlon = msg["Message"]["PositionReport"]["Longitude"]
                    sog_raw = msg["Message"]["PositionReport"].get("Sog", 0)
                    vsog = float(sog_raw)
                    if vsog > 50: vsog = vsog / 10.0 
                    vname = msg["MetaData"]["ShipName"].strip()
                    if not vname: vname = f"Unknown (MMSI: {mmsi})"
                    vessels[mmsi] = {"lat": vlat, "lon": vlon, "name": vname, "sog": vsog}
                    if len(vessels) >= max_vessels: break
            except: break
        ws.close()
    except: pass 
    return list(vessels.values())

# =============================================================================
# BÖLÜM 2: GİRDİLER (SIDEBAR)
# =============================================================================
st.sidebar.success("✅ V9.1 Digital Twin Active")
st.sidebar.header("⚙️ 1. Gemi Tasarım Bilgileri")
dwt = st.sidebar.number_input("Deadweight (DWT) [Ton]", value=105000.0)
beam = st.sidebar.number_input("Gemi Genişliği (Beam) [m]", value=42.8)
d_speed = st.sidebar.number_input("Tasarım Hızı [Knot]", value=24.0)
d_draft = st.sidebar.number_input("Tasarım Draftı [m]", value=14.5)
d_cons = st.sidebar.number_input("Tasarım Tüketimi [Ton/Gün]", value=225.0)

st.sidebar.markdown("---")
st.sidebar.header("📡 2. Uydu & AIS Bağlantıları")
api_key_global = st.sidebar.text_input("OpenWeather API Key:", type="password")
ais_api_key = st.sidebar.text_input("AISStream.io API Key:", type="password")

st.sidebar.markdown("---")
st.sidebar.header("🏗️ 3. Liman & Operasyon Verileri")
port_handling_time = st.sidebar.number_input("Ortalama Elleçleme (Saat/Gemi)", value=12.0, step=1.0)
port_terminals = st.sidebar.number_input("Aktif Rıhtım Sayısı", value=3, min_value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.header("🔬 4. Dijital İkiz Parametreleri")
months_drydock = st.sidebar.slider("Havuzdan Sonra Geçen Süre (Ay)", 0, 60, 12)
selected_fuel = st.sidebar.selectbox("Makine Yakıt Tipi", list(FUEL_DATA.keys()))
eu_ets_tax = st.sidebar.number_input("EU ETS Karbon Vergisi (€/Ton CO2)", value=85.0)

fuel_co2_factor = FUEL_DATA[selected_fuel]["co2_factor"]
fuel_price = FUEL_DATA[selected_fuel]["price_per_ton"]
ref_cii = (d_cons * 3.114 * 1_000_000) / (dwt * d_speed * 24)

# =============================================================================
# BÖLÜM 3: OPENCV RÜZGAR ALANI HESAPLAYICI
# =============================================================================
st.header("📷 Rüzgar Alanı Analizi (OpenCV)")
with st.expander("Geminin Transvers (Ön) Cephe Fotoğrafını Yükle", expanded=False):
    col_cv1, col_cv2 = st.columns(2)
    uploaded_file = col_cv1.file_uploader("Gemi Fotoğrafı", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pixel_area = cv2.contourArea(c)
            if pixel_area == 0: pixel_area = w * h * 0.7 
            scale = beam / float(w) if w > 0 else 0
            st.session_state.calc_wind_area = float(pixel_area * (scale ** 2))
            st.success(f"✅ Rüzgar Alanı Hesaplandı: {st.session_state.calc_wind_area:.1f} m²")
st.markdown("---")

# =============================================================================
# BÖLÜM 4: MAKİNE ÖĞRENMESİ (ML)
# =============================================================================
st.header("🔄 Büyük Veri & Makine Öğrenmesi")
with st.expander("📊 Veri Seti Üret ve ML Modelini Eğit", expanded=False):
    sim_days = st.number_input("Gün Sayısı:", min_value=10, value=10000)
    if st.button("🚀 Veri Üret ve Eğit"):
        with st.spinner("Model eğitiliyor..."):
            np.random.seed(42)
            s_spd = np.random.uniform(d_speed*0.4, d_speed*1.05, sim_days)
            s_dft = np.random.uniform(d_draft*0.5, d_draft, sim_days)
            s_bft = np.random.randint(0, 10, sim_days)
            s_mon = np.random.randint(0, 60, sim_days)
            s_fuel = [calculate_instant_fuel(s_spd[i], s_dft[i], st.session_state.calc_wind_area, s_bft[i], d_speed, d_draft, d_cons, beam, s_mon[i]) for i in range(sim_days)]
            df = pd.DataFrame({'Hız (Knot)': s_spd, 'Draft (m)': s_dft, 'Beaufort': s_bft, 'Havuz Sonrası (Ay)': s_mon, 'Yakıt': s_fuel})
            X = df[['Hız (Knot)', 'Draft (m)', 'Beaufort', 'Havuz Sonrası (Ay)']]
            y = df['Yakıt']
            X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2)
            model = RandomForestRegressor(n_estimators=50).fit(X_t, y_t)
            st.session_state.ai_model = model
            st.success(f"✅ Model Eğitildi! Skor: %{r2_score(y_v, model.predict(X_v))*100:.2f}")

st.markdown("---")

# =============================================================================
# BÖLÜM 5: NAVİGASYON & OPTİMİZASYON RAPORU (HATA DÜZELTİLDİ)
# =============================================================================
st.header("🗺️ Global Navigasyon & ML Auto-Pilot")
ports = {"Tokyo": (35.6, 139.6), "Shanghai": (31.2, 121.5), "Singapore": (1.3, 103.8), "Rotterdam": (51.9, 4.4), "Istanbul": (41.0, 28.9), "Panama": (9.1, -79.6)}
c_n1, c_n2, c_n3 = st.columns(3)
ship_lat = c_n1.number_input("Lat:", value=35.0)
ship_lon = c_n2.number_input("Lon:", value=15.0)
dest_port = c_n3.selectbox("Hedef:", list(ports.keys()), index=3)
opt_priority = st.slider("Hız Stratejisi:", 0, 100, 50)

if st.button("📡 Analizi Başlat"):
    with st.spinner("Hesaplanıyor..."):
        dest_lat, dest_lon = ports[dest_port]
        route = sr.searoute([ship_lon, ship_lat], [dest_lon, dest_lat])
        route_coords = route["geometry"]["coordinates"]
        dist_nm = sum([haversine_distance(route_coords[i][1], route_coords[i][0], route_coords[i+1][1], route_coords[i+1][0]) for i in range(len(route_coords)-1)])
        
        avg_bft = 4
        if api_key_global:
            _, avg_bft = get_live_weather_by_coords(ship_lat, ship_lon, api_key_global)
            if avg_bft is None: avg_bft = 4
            
        max_speed = d_speed
        chosen_speed = 12.0 + (max_speed - 12.0) * (opt_priority / 100.0)
        
        # --- ML TAHMİNLERİ ---
        if st.session_state.ai_model:
            def predict_fuel(s, d, b, m):
                return st.session_state.ai_model.predict(pd.DataFrame([[s, d, b, m]], columns=['Hız (Knot)', 'Draft (m)', 'Beaufort', 'Havuz Sonrası (Ay)']))[0]
            
            daily_fuel = predict_fuel(chosen_speed, d_draft, avg_bft, months_drydock)
            bad_daily_fuel = predict_fuel(max_speed, d_draft, avg_bft, months_drydock)
        else:
            daily_fuel = calculate_instant_fuel(chosen_speed, d_draft, st.session_state.calc_wind_area, avg_bft, d_speed, d_draft, d_cons, beam, months_drydock)
            bad_daily_fuel = calculate_instant_fuel(max_speed, d_draft, st.session_state.calc_wind_area, avg_bft, d_speed, d_draft, d_cons, beam, months_drydock)
            
        days_on_route = dist_nm / (chosen_speed * 24)
        total_fuel = daily_fuel * days_on_route
        bad_days = dist_nm / (max_speed * 24)
        
        # --- [FIX] NameError Düzeltmesi ---
        bad_total_fuel = bad_daily_fuel * bad_days
        bad_co2 = bad_total_fuel * fuel_co2_factor
        # -------------------------------

        # HARİTA
        fig = go.Figure()
        lons, lats = [c[0] for c in route_coords], [c[1] for c in route_coords]
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode='lines', line=dict(width=4, color='lime')))
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), geo=dict(projection_type="equirectangular"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Seyir ve Optimizasyon Raporu")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mesafe", f"{dist_nm:,.0f} Nm")
        c2.success(f"⛽ ML Yakıt: {total_fuel:,.0f} Ton\n\n⏱️ Süre: {days_on_route:,.1f} Gün")
        c3.warning(f"⚠️ Tam Yol: {bad_total_fuel:,.0f} Ton\n\n💸 Karbon Ver.: €{bad_co2 * eu_ets_tax:,.0f}")

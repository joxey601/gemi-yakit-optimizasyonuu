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
    page_title="Gemi Yakıt & CII Optimizasyonu",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Gemi Performans ve Yakıt Simülatörü V8.8 (AI Auto-Pilot & Live ML)")
st.markdown("**Modüller:** Computer Vision | Big Data | **Machine Learning** | Live Satellite | Geospatial Routing | **Real-Time AIS** | JIT Logistics")
st.markdown("---")

# Session State Başlatmaları
if 'calc_wind_area' not in st.session_state:
    st.session_state.calc_wind_area = 800.0
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None

# =============================================================================
# BÖLÜM 1: AKADEMİK FİZİK MOTORU VE AIS BAĞLANTISI
# =============================================================================

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

def calculate_instant_fuel(speed, draft, wind_area, beaufort, des_spd, des_dft, des_cons, beam):
    v_ship_ms = speed * 0.5144
    p_des_kw = (des_cons * 1000000) / (24 * 175) 
    p_calm_kw = p_des_kw * ((speed / des_spd)**3) * ((draft / des_dft)**(2/3))
    
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

def calculate_cii_grade(fuel_ton, speed_knots, dwt, ref_cii):
    if speed_knots <= 0: return 9999, "E"
    attained_cii = (fuel_ton * 3.114 * 1_000_000) / (dwt * speed_knots * 24)
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
    
    subscribe_message = {
        "APIKey": api_key,
        "BoundingBoxes": bounding_box,
        "FilterMessageTypes": ["PositionReport"]
    }
    
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
                    
                    # [YENİ] CANLI GEMİ HIZINI (SOG) ÇEKİYORUZ!
                    sog_raw = msg["Message"]["PositionReport"].get("Sog", 0)
                    vsog = float(sog_raw)
                    if vsog > 50: vsog = vsog / 10.0 # NMEA format düzeltmesi
                    
                    vname = msg["MetaData"]["ShipName"].strip()
                    if not vname: vname = f"Unknown (MMSI: {mmsi})"
                    
                    vessels[mmsi] = {"lat": vlat, "lon": vlon, "name": vname, "sog": vsog}
                    if len(vessels) >= max_vessels: break
            except websocket.WebSocketTimeoutException:
                break
        ws.close()
    except Exception as e:
        pass 
    return list(vessels.values())

# =============================================================================
# BÖLÜM 2: GİRDİLER (SIDEBAR)
# =============================================================================
st.sidebar.success("✅ ISO 15016 Physics & Sklearn ML Active")
st.sidebar.header("⚙️ 1. Gemi Tasarım Bilgileri")
ship_type = st.sidebar.text_input("Gemi Tipi", value="Container 8000TEU")
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
st.sidebar.header("🏗️ 3. Liman Operasyon Verileri")
port_handling_time = st.sidebar.number_input("Ortalama Elleçleme (Saat/Gemi)", value=12.0, step=1.0)
port_terminals = st.sidebar.number_input("Aktif Rıhtım Sayısı", value=3, min_value=1, step=1)

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
            img_result = img.copy()
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 3)
            pixel_area = cv2.contourArea(c)
            if pixel_area == 0: pixel_area = w * h * 0.7 
            scale = beam / float(w) if w > 0 else 0
            real_area = pixel_area * (scale ** 2)
            st.session_state.calc_wind_area = float(real_area)
            img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
            col_cv1.image(img_rgb, caption="OpenCV Kontur ve Bounding Box")
            col_cv2.success(f"✅ Gemi Genişliği (Beam): {beam} m\n\n🌬️ **Hesaplanan Transvers Rüzgar Alanı:** {real_area:.1f} m²")
        else:
            col_cv2.error("Gemi silueti bulunamadı.")
            st.session_state.calc_wind_area = col_cv2.number_input("Manuel Rüzgar Alanı (m²)", value=st.session_state.calc_wind_area)
    else:
        st.session_state.calc_wind_area = col_cv2.number_input("Manuel Rüzgar Alanı (m²)", value=st.session_state.calc_wind_area)

st.markdown("---")

# =============================================================================
# BÖLÜM 4: BÜYÜK VERİ ÜRETİMİ (BIG DATA) & MAKİNE ÖĞRENMESİ
# =============================================================================
st.header("🔄 Büyük Veri Simülasyonu & Çevresel Regülasyon (CII)")

with st.expander("📊 Sanal Sefer Verisi Üret ve ML Modelini Eğit", expanded=False):
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        sim_days = st.number_input("Simüle Edilecek Gün Sayısı (ML Eğitimi İçin):", min_value=10, max_value=20000, value=10000, step=100)
        sim_wind_area = st.number_input("Gemi Rüzgar Alanı (m²):", value=float(st.session_state.calc_wind_area))
        
        if st.button("🚀 Veri Setini Üret ve Yapay Zekayı Eğit"):
            with st.spinner(f"Arka planda {sim_days} günlük okyanus simülasyonu çalıştırılıyor ve model eğitiliyor..."):
                np.random.seed(42) 
                sim_speeds = np.random.uniform(d_speed * 0.4, d_speed * 1.05, sim_days)
                sim_drafts = np.random.uniform(d_draft * 0.5, d_draft, sim_days)
                sim_bfts = np.random.randint(0, 10, sim_days) 
                
                sim_fuels, sim_ciis, sim_grades = [], [], []
                for i in range(sim_days):
                    fuel = calculate_instant_fuel(sim_speeds[i], sim_drafts[i], sim_wind_area, sim_bfts[i], d_speed, d_draft, d_cons, beam)
                    sim_fuels.append(fuel)
                    cii_val, grade = calculate_cii_grade(fuel, sim_speeds[i], dwt, ref_cii)
                    sim_ciis.append(cii_val)
                    sim_grades.append(grade)
                    
                df_sim = pd.DataFrame({
                    "Hız (Knot)": np.round(sim_speeds, 1),
                    "Draft (m)": np.round(sim_drafts, 1), 
                    "Beaufort": sim_bfts,
                    "Günlük Yakıt (Ton)": np.round(sim_fuels, 1), 
                    "CII Değeri": np.round(sim_ciis, 2), 
                    "IMO Karne": sim_grades
                })
                st.session_state.df_sim = df_sim
                
                # ML Eğitimi
                X = df_sim[['Hız (Knot)', 'Draft (m)', 'Beaufort']]
                y = df_sim['Günlük Yakıt (Ton)']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                
                y_pred = rf_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.session_state.ai_model = rf_model
                st.session_state.ml_metrics = {"mae": mae, "r2": r2}
                
                st.success(f"✅ {sim_days} günlük veri üretildi ve Makine Öğrenmesi başarıyla eğitildi!")

    with col_sim2:
        if st.session_state.ai_model is not None:
            st.info(f"**🤖 Makine Öğrenmesi Başarı Metrikleri (Sınav Sonucu)**\n\n"
                    f"**R² Skoru (Doğruluk Oranı):** %{st.session_state.ml_metrics['r2'] * 100:.2f}\n\n"
                    f"**Hata Payı (MAE):** ±{st.session_state.ml_metrics['mae']:.2f} Ton/Gün\n\n"
                    f"*Not: Navigasyon ve JIT hesaplamaları artık formüllerle değil, bu AI modeli ile yapılmaktadır.*")

st.markdown("---")

# =============================================================================
# BÖLÜM 5: GLOBAL NAVİGASYON & CANLI AIS + ML ENTEGRASYONU
# =============================================================================
st.header("🗺️ Global Navigasyon & ML Auto-Pilot")

ports = {
    "Tokyo": (35.6, 139.6), "Shanghai": (31.2, 121.5), "Singapore": (1.3, 103.8),
    "Colombo": (6.9, 79.8), "Suez": (31.2, 32.3), "Istanbul": (41.0, 28.9),
    "Rotterdam": (51.9, 4.4), "New York": (40.7, -74.0), "Panama": (9.1, -79.6)
}

col_nav1, col_nav2, col_nav3 = st.columns(3)
ship_lat = col_nav1.number_input("Geminin Enlemi (Lat):", value=35.0000, format="%.4f")
ship_lon = col_nav2.number_input("Geminin Boylamı (Lon):", value=15.0000, format="%.4f")
dest_port = col_nav3.selectbox("Varış Limanı (Hedef):", list(ports.keys()), index=6) 

dest_lat, dest_lon = ports[dest_port]

st.markdown("#### ⚖️ Kaptan Kararı (Manuel Öncelik)")
opt_priority = st.slider("Eğer ML Modeli Aktif Değilse Geçerli Olacak Seyir Stratejisi:", min_value=0, max_value=100, value=50, step=10)

if st.button("📡 AIS Verisini Çek ve Otonom ML Analizini Başlat"):
    with st.spinner("Okyanus verileri, AIS Trafiği ve ML Rota Optimizasyonu çalışıyor..."):
        
        origin, destination = [ship_lon, ship_lat], [dest_lon, dest_lat]
        
        try:
            route = sr.searoute(origin, destination)
            if route is None:
                st.error("Rota bulunamadı. Koordinatları kontrol edin.")
                st.stop()
            route_coords = route["geometry"]["coordinates"] 
        except Exception as e:
            st.error(f"🚨 HATA DETAYI: {str(e)}")
            st.stop()
            
        total_distance_nm = sum([haversine_distance(route_coords[i][1], route_coords[i][0], route_coords[i+1][1], route_coords[i+1][0]) for i in range(len(route_coords)-1)])
            
        num_samples = 8 
        step = max(1, len(route_coords) // num_samples)
        route_segments, avg_bft = [], 3 
        
        if api_key_global:
            bft_sum, valid_samples = 0, 0
            my_bar = st.progress(0.0, text="Hava durumu taraması...")
            
            for idx, i in enumerate(range(0, len(route_coords), step)):
                end_idx = min(i + step + 1, len(route_coords)) 
                segment_coords = route_coords[i:end_idx]
                lon_s, lat_s = segment_coords[0]
                _, bft = get_live_weather_by_coords(lat_s, lon_s, api_key_global)
                if bft is None: bft = 3
                bft_sum += bft
                valid_samples += 1
                seg_color = 'red' if bft >= 6 else 'lime'
                route_segments.append({'coords': segment_coords, 'color': seg_color, 'bft': bft})
                my_bar.progress(min(1.0, (idx + 1) / num_samples)) 
            if valid_samples > 0: avg_bft = round(bft_sum / valid_samples)
            my_bar.empty()
        else:
            route_segments.append({'coords': route_coords, 'color': 'lime', 'bft': 3})
            
        # CANLI GEMİLERİ ÇEKME VE HIZ ANALİZİ
        live_vessels = []
        port_vessels_count = 0
        fleet_speeds = []
        
        if ais_api_key:
            st.toast("📡 AIS Radarı ML entegrasyonu için aktif ediliyor...")
            live_vessels = get_ais_snapshot_route(ais_api_key, route_coords)
            if live_vessels:
                st.toast(f"✅ Rota boyunca {len(live_vessels)} gerçek gemi ve hız (SOG) verileri çekildi!")
                for v in live_vessels:
                    if v['sog'] > 5.0 and v['sog'] < 30.0: # Seyir halindeki mantıklı hızları al
                        fleet_speeds.append(v['sog'])
                    
                    dist_to_port = haversine_distance(dest_lat, dest_lon, v['lat'], v['lon'])
                    if dist_to_port <= 30.0:
                        port_vessels_count += 1
        
        # --- ML MODELİ AKTİF Mİ KONTROLÜ VE YAKIT HESABI ---
        max_speed = d_speed
        eco_speed = max(10.0, d_speed * 0.5)
        chosen_speed = eco_speed + ((max_speed - eco_speed) * (opt_priority / 100.0))
        w_area = st.session_state.calc_wind_area
        
        # JIT Bekleme ve Hız Hesaplaması
        jit_wait_hours = (port_vessels_count * port_handling_time) / port_terminals
        days_on_route = total_distance_nm / (chosen_speed * 24)
        jit_target_days = days_on_route + (jit_wait_hours / 24.0)
        jit_speed = total_distance_nm / (jit_target_days * 24)
        if jit_speed < 8.0: jit_speed = 8.0
            
        if st.session_state.ai_model is not None:
            st.toast("🤖 Yakıt optimizasyonları Scikit-Learn Modeli ile yapılıyor...")
            
            # AI ML ile Yakıt Tahminleri
            daily_fuel = st.session_state.ai_model.predict(pd.DataFrame({'Hız (Knot)': [chosen_speed], 'Draft (m)': [d_draft], 'Beaufort': [avg_bft]}))[0]
            bad_daily_fuel = st.session_state.ai_model.predict(pd.DataFrame({'Hız (Knot)': [max_speed], 'Draft (m)': [d_draft], 'Beaufort': [avg_bft]}))[0]
            jit_daily_fuel = st.session_state.ai_model.predict(pd.DataFrame({'Hız (Knot)': [jit_speed], 'Draft (m)': [d_draft], 'Beaufort': [avg_bft]}))[0]
            
            # Rakip Filo Analizi (Canlı AIS hızlarının ML ile yakıt tahmini)
            avg_fleet_speed = np.mean(fleet_speeds) if len(fleet_speeds) > 0 else chosen_speed
            fleet_daily_fuel = st.session_state.ai_model.predict(pd.DataFrame({'Hız (Knot)': [avg_fleet_speed], 'Draft (m)': [d_draft], 'Beaufort': [avg_bft]}))[0]
            fleet_total_fuel = fleet_daily_fuel * (total_distance_nm / (avg_fleet_speed * 24))
            
        else:
            daily_fuel = calculate_instant_fuel(chosen_speed, d_draft, w_area, avg_bft, d_speed, d_draft, d_cons, beam)
            bad_daily_fuel = calculate_instant_fuel(max_speed, d_draft, w_area, avg_bft, d_speed, d_draft, d_cons, beam)
            jit_daily_fuel = calculate_instant_fuel(jit_speed, d_draft, w_area, avg_bft, d_speed, d_draft, d_cons, beam)
            avg_fleet_speed = np.mean(fleet_speeds) if len(fleet_speeds) > 0 else chosen_speed
            fleet_daily_fuel = calculate_instant_fuel(avg_fleet_speed, d_draft, w_area, avg_bft, d_speed, d_draft, d_cons, beam)
            fleet_total_fuel = fleet_daily_fuel * (total_distance_nm / (avg_fleet_speed * 24))

        total_fuel = daily_fuel * days_on_route
        bad_days = total_distance_nm / (max_speed * 24)
        bad_total_fuel = bad_daily_fuel * bad_days
        jit_total_fuel = jit_daily_fuel * (total_distance_nm / (jit_speed * 24))
        jit_savings = total_fuel - jit_total_fuel
        
        _, ai_cii_grade = calculate_cii_grade(total_fuel / days_on_route, chosen_speed, dwt, ref_cii)
        _, bad_cii_grade = calculate_cii_grade(bad_daily_fuel, max_speed, dwt, ref_cii)
        
        # --- HARİTA ÇİZİMİ ---
        fig = go.Figure()

        for seg in route_segments:
            lon_list = [c[0] for c in seg['coords']]
            lat_list = [c[1] for c in seg['coords']]
            fig.add_trace(go.Scattergeo(lon=lon_list, lat=lat_list, mode='lines', line=dict(width=4, color=seg['color']), hoverinfo='text', text=f"Hava: {seg['bft']} Bft", showlegend=False))
            
        fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode='lines', line=dict(color='lime', width=4), name='Güvenli Rota (<6 Bft)'))
        fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode='lines', line=dict(color='red', width=4), name='Fırtınalı Rota (≥6 Bft)'))

        if live_vessels:
            v_lats = [v['lat'] for v in live_vessels]
            v_lons = [v['lon'] for v in live_vessels]
            v_names = [f"{v['name']} (Hız: {v['sog']} Kn)" for v in live_vessels]
            
            fig.add_trace(go.Scattergeo(
                lon=v_lons, lat=v_lats, mode='markers',
                marker=dict(size=5, color='blue', symbol='circle', opacity=0.6),
                text=v_names, hoverinfo='text', name='Canlı AIS Trafiği'
            ))

        fig.add_trace(go.Scattergeo(lon=[ship_lon], lat=[ship_lat], mode='markers+text', marker=dict(size=14, color='orange', symbol='triangle-up'), text=["📍 ÇIKIŞ"], textposition="top right", name="Gemi Konumu"))
        fig.add_trace(go.Scattergeo(lon=[dest_lon], lat=[dest_lat], mode='markers+text', marker=dict(size=14, color='blue', symbol='star'), text=[f"🏁 {dest_port}"], textposition="bottom center", name="Varış Limanı"))

        fig.update_layout(
            title_text=f'Gerçek Deniz Yolu Navigasyonu (Ortalama Hava: {avg_bft} Bft)',
            showlegend=True,
            dragmode='pan',
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
            geo=dict(showland=True, landcolor="rgb(243, 243, 243)", showocean=True, oceancolor="rgb(204, 229, 255)", showcountries=True, countrycolor="rgb(204, 204, 204)", projection_type="equirectangular", fitbounds="locations"),
            height=500, margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        
        # --- ML DESTEKLİ JIT PANELİ ---
        st.markdown("### 🧠 Yapay Zeka JIT Karar Destek Mekanizması (ML Entegreli)")
        if not ais_api_key:
            st.warning("⚠️ **Veri Bekleniyor:** Liman yoğunluğunu analiz edip hız tavsiyesi verebilmemiz için lütfen sol menüden AISStream API şifresini girin.")
        else:
            if len(fleet_speeds) > 0:
                st.info(f"🛰️ **AIS Rota Analizi:** Okyanus güzergahınızdaki diğer gemilerin anlık ortalama hızı **{avg_fleet_speed:.1f} knot** olarak ölçülmüştür. Sizin otonom ML sisteminiz bu veriyi de değerlendiriyor.")
                
            if port_vessels_count > 0:
                if jit_savings > 0:
                    st.error(f"⚠️ **DİKKAT LİMAN YOĞUNLUĞU:** {dest_port} limanı etrafında şu an **{port_vessels_count} adet gemi** var. Tahmini bekleme süreniz **{jit_wait_hours:.0f} saat** olacaktır.")
                    st.success(f"💡 **AI KAPTAN TAVSİYESİ:** Hızınızı **{jit_speed:.1f} knot**'a düşürün. Limana {jit_wait_hours:.0f} saat geç vararak iskeleye sıfır bekleme ile yanaşabilir ve **{jit_savings:,.0f} Ton** fazladan yakıt tasarrufu sağlayabilirsiniz!")
                else:
                    st.info(f"ℹ️ {dest_port} limanında trafik var ancak mevcut hızınız ({chosen_speed:.1f} knot) ML algoritmasına göre zaten optimum aralıkta.")
            else:
                st.success(f"✅ **LİMAN AÇIK:** AIS verilerine göre şu an {dest_port} limanında bekleyen gemi tespit edilmedi.")
        st.markdown("---")

        st.subheader("📊 Seyir ve ML Optimizasyon Raporu")
        c1, c2, c3 = st.columns(3)
        c1.info(f"📏 **Toplam Mesafe:** {total_distance_nm:,.0f} Nm\n\n🌬️ **Okyanus Ortalaması:** {avg_bft} Beaufort")
        c2.success(f"⛽ **ML Yakıt Tüketimi:** {total_fuel:,.0f} Ton\n\n⏱️ **Varış Süresi:** {days_on_route:,.1f} Gün\n\n🚀 **Uygulanan Hız:** {chosen_speed:.1f} Knot\n\n🏆 **Sefer CII Karnesi:** Sınıf {ai_cii_grade}")
        
        if total_fuel < bad_total_fuel:
            c3.warning(f"⚠️ **Tam Yol (Max Hız):**\n\n⛽ Tüketim: {bad_total_fuel:,.0f} Ton\n⏱️ Süre: {bad_days:,.1f} Gün\n\n📉 **Kötü Senaryo Karnesi:** Sınıf {bad_cii_grade}")
        else:
            c3.success(f"✨ Maksimum hız önceliği seçtiniz. AI Gemi tasarruf yapmadan hedefe en kısa sürede ulaşıyor.")

        st.markdown("---")
        st.subheader("📈 ML (Makine Öğrenmesi) Hız-Yakıt Optimizasyon Eğrisi")
        col_g1, col_g2 = st.columns(2)

        fig_bar = go.Figure(data=[go.Bar(name='Yakıt Tüketimi (Ton)', x=['Tam Yol (Manuel)', 'Sektör Ortalaması (AIS)', 'AI Optimize (JIT)'], y=[bad_total_fuel, fleet_total_fuel, jit_total_fuel], text=[f"{bad_total_fuel:,.0f} Ton", f"{fleet_total_fuel:,.0f} Ton", f"{jit_total_fuel:,.0f} Ton"], textposition='auto', marker_color=['#e74c3c', '#f39c12', '#2ecc71'])])
        fig_bar.update_layout(title='Makine Öğrenmesi Tabanlı Filo Karşılaştırması', yaxis_title='Toplam Yakıt (Ton)', height=400)
        col_g1.plotly_chart(fig_bar, use_container_width=True)

        speeds_array = np.linspace(eco_speed, max_speed, 20)
        
        if st.session_state.ai_model is not None:
            fuels_array = [st.session_state.ai_model.predict(pd.DataFrame({'Hız (Knot)': [s], 'Draft (m)': [d_draft], 'Beaufort': [avg_bft]}))[0] * (total_distance_nm / (s * 24)) for s in speeds_array]
        else:
            fuels_array = [calculate_instant_fuel(s, d_draft, w_area, avg_bft, d_speed, d_draft, d_cons, beam) * (total_distance_nm / (s * 24)) for s in speeds_array]
            
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=speeds_array, y=fuels_array, mode='lines', name='ML Hız-Yakıt Karakteristiği', line=dict(color='#3498db', width=3)))
        fig_curve.add_trace(go.Scatter(x=[max_speed], y=[bad_total_fuel], mode='markers+text', name='Tam Yol', marker=dict(color='#e74c3c', size=12, symbol='x'), text=['❌ Tam Yol'], textposition='top right'))
        fig_curve.add_trace(go.Scatter(x=[jit_speed], y=[jit_total_fuel], mode='markers+text', name='JIT AI Optimizasyonu', marker=dict(color='#2ecc71', size=14), text=['✅ JIT Optimizasyonu'], textposition='bottom right'))
        
        fig_curve.update_layout(title='Yapay Zeka (Scikit-Learn) Optimizasyon Eğrisi', xaxis_title='Gemi Hızı (Knot)', yaxis_title='Toplam Sefer Yakıtı (Ton)', height=400, dragmode='pan')
        col_g2.plotly_chart(fig_curve, use_container_width=True, config={'scrollZoom': True})

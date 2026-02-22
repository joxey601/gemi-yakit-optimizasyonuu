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

# =============================================================================
# SAYFA AYARLARI VE ANA PANEL
# =============================================================================
st.set_page_config(
    page_title="Gemi Yakıt & CII Optimizasyonu V9.6",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Gemi Performans ve Yakıt Simülatörü V9.6 (Master Edition)")
st.markdown("**Modüller:** Computer Vision | Big Data | Machine Learning | Live AIS Radar | JIT Decision Support | Predictive Maintenance | EU ETS | Dynamic Weather Routing")
st.markdown("---")

# Session State Başlatmaları (Global Değişkenler)
if 'calc_wind_area' not in st.session_state:
    st.session_state.calc_wind_area = 800.0
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None

# =============================================================================
# BÖLÜM 1: AKADEMİK FİZİK MOTORU & DİJİTAL İKİZ KÜTÜPHANESİ
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
    """ISO 15016 ve ITTC Tabanlı Gelişmiş Direnç ve Yakıt Hesaplayıcı"""
    v_ship_ms = speed * 0.5144
    p_des_kw = (des_cons * 1000000) / (24 * 175) 
    
    # 1. Kestirimci Bakım: Biofouling (Kirlenme) Etkisi (Aylık %1.5 artış)
    bio_penalty = 1.0 + (months_since_drydock * 0.015) 
    p_calm_kw = p_des_kw * ((speed / des_spd)**3) * ((draft / des_dft)**(2/3)) * bio_penalty
    
    # 2. Rüzgar Direnci (ISO 15016)
    wind_speed_ms = 0.836 * (beaufort ** 1.5)
    v_rel_ms = v_ship_ms + wind_speed_ms 
    rho_air = 1.225 
    c_aa = 0.8 
    r_aa_newton = 0.5 * rho_air * c_aa * wind_area * (v_rel_ms ** 2)
    p_wind_kw = (r_aa_newton * v_ship_ms) / 1000
    
    # 3. Dalga Direnci (ITTC / STAwave-1)
    rho_water, g = 1025, 9.81
    h_s = 0.2 * (beaufort ** 2) 
    l_wl = beam * 6.5 
    r_wave_newton = (1/16) * rho_water * g * (h_s ** 2) * beam * math.sqrt(beam / l_wl)
    p_wave_kw = (r_wave_newton * v_ship_ms) / 1000
    
    p_total_kw = p_calm_kw + p_wind_kw + p_wave_kw
    
    # Dinamik SFOC Eğrisi
    mcr_kw = p_des_kw * 1.1 
    load = max(0.1, p_total_kw / mcr_kw)
    sfoc_dyn = 175 * (1 + 0.5 * (load - 0.75)**2)
    
    return (p_total_kw * sfoc_dyn * 24) / 1000000

def calculate_cii_grade(fuel_ton, speed_knots, dwt, ref_cii, fuel_co2_factor):
    if speed_knots <= 0: return 9999, "E"
    attained_cii = (fuel_ton * fuel_co2_factor * 1_000_000) / (dwt * speed_knots * 24)
    ratio = attained_cii / ref_cii
    if ratio < 0.83: return attained_cii, "A"
    elif ratio < 0.94: return attained_cii, "B"
    elif ratio < 1.06: return attained_cii, "C"
    elif ratio < 1.19: return attained_cii, "D"
    else: return attained_cii, "E"

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3440.065
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def get_ais_snapshot_route(api_key, route_coords, margin_deg=5.0, timeout=5, max_vessels=1500):
    lats, lons = [c[1] for c in route_coords], [c[0] for c in route_coords]
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
                    vlat, vlon = msg["Message"]["PositionReport"]["Latitude"], msg["Message"]["PositionReport"]["Longitude"]
                    sog_raw = msg["Message"]["PositionReport"].get("Sog", 0)
                    vsog = float(sog_raw) / 10.0 if float(sog_raw) > 50 else float(sog_raw)
                    vessels[mmsi] = {"lat": vlat, "lon": vlon, "name": msg["MetaData"]["ShipName"].strip() or f"MMSI:{mmsi}", "sog": vsog}
                    if len(vessels) >= max_vessels: break
            except: break
        ws.close()
    except: pass 
    return list(vessels.values())

# =============================================================================
# BÖLÜM 2: SIDEBAR GİRDİLERİ
# =============================================================================
st.sidebar.success("✅ V9.6 Master Engine Active")
st.sidebar.header("⚙️ 1. Gemi Tasarım Bilgileri")
dwt = st.sidebar.number_input("Deadweight (DWT) [Ton]", value=105000.0)
beam = st.sidebar.number_input("Gemi Genişliği (Beam) [m]", value=42.8)
d_speed = st.sidebar.number_input("Tasarım Hızı [Knot]", value=24.0)
d_draft = st.sidebar.number_input("Tasarım Draftı [m]", value=14.5)
d_cons = st.sidebar.number_input("Tasarım Tüketimi [Ton/Gün]", value=225.0)

st.sidebar.markdown("---")
st.sidebar.header("📡 2. API Bağlantıları")
api_key_global = st.sidebar.text_input("OpenWeather API Key:", type="password")
ais_api_key = st.sidebar.text_input("AISStream.io API Key:", type="password")

st.sidebar.markdown("---")
st.sidebar.header("🏗️ 3. Liman & JIT Verileri")
port_handling_time = st.sidebar.number_input("Elleçleme Hızı (Saat/Gemi)", value=12.0)
port_terminals = st.sidebar.number_input("Aktif Rıhtım Sayısı", value=3, min_value=1)

st.sidebar.markdown("---")
st.sidebar.header("🔬 4. Dijital İkiz Parametreleri")
months_drydock = st.sidebar.slider("Havuzlama Sonrası Süre (Ay)", 0, 60, 12)
selected_fuel = st.sidebar.selectbox("Yakıt Tipi Seçimi", list(FUEL_DATA.keys()))
eu_ets_tax = st.sidebar.number_input("EU ETS Karbon Vergisi (€/Ton)", value=85.0)

fuel_co2_factor = FUEL_DATA[selected_fuel]["co2_factor"]
fuel_price = FUEL_DATA[selected_fuel]["price_per_ton"]
ref_cii = (d_cons * 3.114 * 1_000_000) / (dwt * d_speed * 24)

# =============================================================================
# BÖLÜM 3: OPENCV RÜZGAR ALANI ANALİZİ
# =============================================================================
st.header("📷 OpenCV Rüzgar Alanı Analizi")
with st.expander("Gemi Transvers Cephe Fotoğrafı Yükle", expanded=False):
    col_c1, col_c2 = st.columns(2)
    up_file = col_c1.file_uploader("Fotoğraf Seç", type=['png', 'jpg', 'jpeg'])
    if up_file:
        img = cv2.imdecode(np.asarray(bytearray(up_file.read()), dtype=np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            st.session_state.calc_wind_area = float(cv2.contourArea(c) * ((beam/w)**2))
            col_c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Orijinal")
            col_c2.success(f"🌬️ Otomatik Hesaplanan Alan: {st.session_state.calc_wind_area:.1f} m²")

st.markdown("---")

# =============================================================================
# BÖLÜM 4: BÜYÜK VERİ & 4D MACHINE LEARNING
# =============================================================================
st.header("🔄 Big Data & Machine Learning (Scikit-Learn)")
with st.expander("📊 10,000 Verilik AI Modelini Eğit", expanded=False):
    col_ml1, col_ml2 = st.columns([1, 2])
    with col_ml1:
        sim_count = st.number_input("Veri Miktarı:", value=10000)
        if st.button("🧠 AI Modelini Ateşle"):
            with st.spinner("Yapay zeka okyanusu ezberliyor..."):
                np.random.seed(42)
                s_v = np.random.uniform(d_speed*0.4, d_speed*1.05, sim_count)
                s_d = np.random.uniform(d_draft*0.5, d_draft, sim_count)
                s_b = np.random.randint(0, 11, sim_count)
                s_m = np.random.randint(0, 61, sim_count)
                s_f = [calculate_instant_fuel(s_v[i], s_d[i], st.session_state.calc_wind_area, s_b[i], d_speed, d_draft, d_cons, beam, s_m[i]) for i in range(sim_count)]
                
                df = pd.DataFrame({'Hız': s_v, 'Draft': s_d, 'Bft': s_b, 'Ay': s_m, 'Yakıt': s_f})
                X = df[['Hız', 'Draft', 'Bft', 'Ay']]
                y = df['Yakıt']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
                st.session_state.ai_model = model
                st.session_state.ml_res = {"mae": mean_absolute_error(y_test, model.predict(X_test)), "r2": r2_score(y_test, model.predict(X_test))}
                st.success("✅ Model Eğitimi Başarılı!")

    with col_ml2:
        if st.session_state.ai_model:
            st.info(f"📈 **Doğruluk:** %{st.session_state.ml_res['r2']*100:.2f} | **Hata:** ±{st.session_state.ml_res['mae']:.2f} Ton/Gün")
            if 'df' in locals(): st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# =============================================================================
# BÖLÜM 5: GLOBAL NAVİGASYON, WEATHER ROUTING & JIT RADAR
# =============================================================================
st.header("🗺️ Global Navigasyon & AI Auto-Pilot")
ports = {"Tokyo": (35.6, 139.6), "Shanghai": (31.2, 121.5), "Singapore": (1.3, 103.8), "Rotterdam": (51.9, 4.4), "Istanbul": (41.0, 28.9), "Panama": (9.1, -79.6), "New York": (40.7, -74.0)}
c_n1, c_n2, c_n3 = st.columns(3)
ship_lat, ship_lon = c_n1.number_input("Mevcut Lat:", value=35.0), c_n2.number_input("Mevcut Lon:", value=15.0)
dest_port = c_n3.selectbox("Varış Limanı:", list(ports.keys()), index=3)
strategy = st.slider("Hız Önceliği (%):", 0, 100, 50)

if st.button("📡 Rota Analizini ve Canlı AIS Taramasını Başlat"):
    with st.spinner("Veriler işleniyor..."):
        dest_lat, dest_lon = ports[dest_port]
        try:
            route_data = sr.searoute([ship_lon, ship_lat], [dest_lon, dest_lat])
            coords = route_data["geometry"]["coordinates"]
        except Exception as e:
            st.error("Rota çizilemedi, lütfen koordinatları kontrol edin.")
            st.stop()
        
        # Dinamik Weather Routing & Mesafe Hesabı
        segments, total_dist_nm, bft_sum, wr_active, wr_saved = [], 0, 0, False, 0
        step = max(1, len(coords) // 10)
        
        for i in range(0, len(coords)-1, step):
            end = min(i + step + 1, len(coords))
            seg = coords[i:end]
            d = sum([haversine_distance(seg[k][1], seg[k][0], seg[k+1][1], seg[k+1][0]) for k in range(len(seg)-1)])
            bft = 4
            if api_key_global:
                _, bft_val = get_live_weather_by_coords(seg[0][1], seg[0][0], api_key_global)
                bft = bft_val or 4
            
            # Weather Routing (Mor Çizgi Mantığı)
            if bft >= 7:
                f_detour = calculate_instant_fuel(d_speed, d_draft, st.session_state.calc_wind_area, 4, d_speed, d_draft, d_cons, beam, months_drydock) * (d*1.15/(d_speed*24))
                f_storm = calculate_instant_fuel(d_speed, d_draft, st.session_state.calc_wind_area, bft, d_speed, d_draft, d_cons, beam, months_drydock) * (d/(d_speed*24))
                if f_detour < f_storm:
                    wr_active, bft, d, wr_saved = True, 4, d*1.15, wr_saved+(f_storm-f_detour)
                    seg = [[c[0]+0.85, c[1]-0.85] for c in seg]
                    segments.append({'c': seg, 'color': 'fuchsia'})
                else: segments.append({'c': seg, 'color': 'red'})
            else: segments.append({'c': seg, 'color': 'lime'})
            total_dist_nm += d
            bft_sum += bft
            
        avg_bft = round(bft_sum / len(segments))

        # AIS & JIT Analizi
        live_ships, port_density, fleet_speeds = [], 0, []
        if ais_api_key:
            live_ships = get_ais_snapshot_route(ais_api_key, coords)
            for s in live_ships:
                if 5 < s['sog'] < 35: fleet_speeds.append(s['sog'])
                if haversine_distance(dest_lat, dest_lon, s['lat'], s['lon']) <= 30: port_density += 1

        # Hız Optimizasyon Hesapları
        max_v = d_speed
        chosen_v = 10.0 + (max_v - 10.0) * (strategy / 100.0)
        days = total_dist_nm / (chosen_v * 24)
        jit_wait_h = (port_density * port_handling_time) / port_terminals
        jit_v = max(8.0, total_dist_nm / ((days + jit_wait_h/24) * 24))
        
        # [FIX: NameError Düzeltmesi Burada Yapıldı]
        bad_days = total_dist_nm / (max_v * 24) 

        def predict_final(v, b, m):
            if st.session_state.ai_model:
                return st.session_state.ai_model.predict(pd.DataFrame([[v, d_draft, b, m]], columns=['Hız', 'Draft', 'Bft', 'Ay']))[0]
            return calculate_instant_fuel(v, d_draft, st.session_state.calc_wind_area, b, d_speed, d_draft, d_cons, beam, m)

        total_f = predict_final(chosen_v, avg_bft, months_drydock) * days
        bad_f = predict_final(max_v, avg_bft, months_drydock) * bad_days
        jit_f = predict_final(jit_v, avg_bft, months_drydock) * (total_dist_nm / (jit_v * 24))
        clean_f = predict_final(chosen_v, avg_bft, 0) * days
        
        _, ai_cii_grade = calculate_cii_grade(total_f / days, chosen_v, dwt, ref_cii, fuel_co2_factor)
        
        # Harita Çizimi
        fig = go.Figure()
        for s in segments:
            fig.add_trace(go.Scattergeo(lon=[c[0] for c in s['c']], lat=[c[1] for c in s['c']], mode='lines', line=dict(width=5, color=s['color'])))
        if live_ships:
            fig.add_trace(go.Scattergeo(lon=[v['lon'] for v in live_ships], lat=[v['lat'] for v in live_ships], mode='markers', marker=dict(size=6, color='blue'), text=[v['name'] for v in live_ships]))
        
        fig.add_trace(go.Scattergeo(lon=[ship_lon], lat=[ship_lat], mode='markers+text', marker=dict(size=14, color='orange', symbol='triangle-up'), text=["📍 ÇIKIŞ"], textposition="top right"))
        fig.add_trace(go.Scattergeo(lon=[dest_lon], lat=[dest_lat], mode='markers+text', marker=dict(size=14, color='blue', symbol='star'), text=[f"🏁 {dest_port}"], textposition="bottom center"))
        
        fig.update_layout(dragmode='pan', geo=dict(projection_type="equirectangular", showland=True, landcolor="#f0f0f0"), height=600, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        # --- DEV STRATEJİK RAPOR PANELİ ---
        st.markdown("### 🔬 Dijital İkiz: Karar Destek Mekanizması")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.info(f"📏 **Rota:** {total_dist_nm:,.0f} Nm\n\n🌬️ **Hava:** {avg_bft} Beaufort")
            if wr_active: st.success(f"🌪️ **AI Rota:** Fırtınadan kaçılarak {wr_saved:,.0f} Ton yakıt kurtarıldı!")
        with p2:
            st.warning(f"🌱 **Emisyon:** {selected_fuel}\n\n💸 **ETS Vergisi:** €{total_f * fuel_co2_factor * eu_ets_tax:,.0f}")
        with p3:
            bio_extra = total_f - clean_f
            st.error(f"🦠 **Kirlenme:** {months_drydock} Ay\n\n💰 **Ek Maliyet:** ${bio_extra * fuel_price:,.0f}")

        if port_density > 0 and total_f > jit_f:
            st.success(f"💡 **AI JIT TAVSİYESİ:** Limanda {port_density} gemi kuyruğu var. Hızınızı **{jit_v:.1f} knot** seviyesine çekerek **{total_f - jit_f:,.0f} Ton** daha tasarruf edebilirsiniz!")

        st.subheader("📊 Seyir ve Finansal Rapor")
        r1, r2, r3 = st.columns(3)
        r1.metric("Tahmini Tüketim", f"{total_f:,.0f} Ton", f"CII: Sınıf {ai_cii_grade}")
        r2.metric("Varış Süresi", f"{days:,.1f} Gün")
        
        # Hata düzeldiği için burası artık patlamayacak
        bad_co2_total = bad_f * fuel_co2_factor
        r3.metric("Kötü Senaryo (Tam Yol)", f"{bad_f:,.0f} Ton", f"Vergi: €{bad_co2_total * eu_ets_tax:,.0f}", delta_color="inverse")

        st.markdown("---")
        st.subheader("📈 ML Yakıt Optimizasyon Eğrisi")
        v_range = np.linspace(10, max_v, 20)
        f_range = [predict_final(vx, avg_bft, months_drydock) * (total_dist_nm/(vx*24)) for vx in v_range]
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=v_range, y=f_range, mode='lines', name='ML Eğrisi', line=dict(color='#3498db', width=3)))
        fig_curve.add_trace(go.Scatter(x=[max_v], y=[bad_f], mode='markers+text', name='Tam Yol', marker=dict(color='#e74c3c', size=12, symbol='x'), text=['❌ Tam Yol'], textposition='top right'))
        fig_curve.add_trace(go.Scatter(x=[jit_v], y=[jit_f], mode='markers+text', name='JIT AI Optimizasyonu', marker=dict(color='#2ecc71', size=14), text=['✅ JIT Optimizasyonu'], textposition='bottom right'))
        fig_curve.update_layout(xaxis_title='Hız (Knot)', yaxis_title='Toplam Sefer Yakıtı (Ton)', height=400)
        st.plotly_chart(fig_curve, use_container_width=True)

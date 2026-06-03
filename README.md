# 🚢 Digital Twin & OPEX Simulator (Vessel AI & CII Optimizer)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-orange.svg)

An advanced, end-to-end maritime SaaS prototype designed to optimize vessel operations, minimize Total Operational Expenses (OPEX), and ensure environmental compliance (CII & EU ETS). By combining physics-based naval architecture principles with 5-Dimensional Machine Learning, this application serves as a true "Digital Twin" for ship management.

## 🌟 Key Features & Modules

### 1. 📷 Computer Vision for Naval Architecture
* Extracts the transverse frontal wind area of the vessel automatically using **OpenCV** contour detection and bounding boxes from uploaded ship profile images.

### 2. 🧠 5D Explainable Machine Learning (SHAP)
* A **Random Forest Regressor** trained on dynamically generated ocean data.
* **5 Dimensions:** Speed, Draft, Weather (Beaufort), Hull Biofouling (Months since drydock), and Vessel Trim.
* Features **SHAP (SHapley Additive exPlanations)** integration to eliminate the "black box" effect, mathematically explaining which operational factors are burning the most fuel.

### 3. 🌍 Commercial Weather Routing & OPEX Trade-off
* Uses the `searoute` algorithm to generate global navigation paths.
* **Smart Detour Logic:** Evaluates storm cells (Bft >= 7) and calculates the trade-off between *punching through the storm* vs. *taking a longer detour*. It factors in not just fuel, but **Daily Charter/Crew Rates** and **EU ETS Carbon Taxes** to make commercial decisions.

### 4. ⚖️ MARPOL ECA Zone Compliance
* Built-in Geofencing for Emission Control Areas (North Sea, Baltic, US East Coast).
* Automatically switches simulated fuel consumption from HFO to compliant MGO (Marine Gas Oil) when entering ECA zones, reflecting accurate financial penalties.

### 5. ⏱️ Just-In-Time (JIT) Arrival & Live AIS
* Connects to **Live AIS Data** (via aisstream.io) to monitor port congestion.
* Calculates JIT speed reduction to eliminate anchor waiting times, reducing both carbon footprint and total OPEX.

### 6. 📡 Engine Control Room (ECR) Live Telemetry
* Simulates real-time IoT sensor data (Main Engine RPM, Exhaust Gas Temperature, Scavenge Air Pressure, Dynamic SFOC) based on AI-optimized speed parameters using **Plotly** dynamic charts.

### 7. 📑 Automated Captain's Voyage Orders
* Generates a formal, downloadable PDF Voyage Order containing optimized routing instructions, speed parameters, and OPEX estimates using the `FPDF` library.

---

## 🛠️ Technology Stack

* **Frontend & Framework:** Streamlit
* **Machine Learning:** Scikit-Learn, SHAP
* **Computer Vision:** OpenCV (opencv-python-headless)
* **Data Manipulation & Math:** Pandas, NumPy, Math
* **Geospatial & Visualization:** Plotly Graph Objects, Searoute
* **APIs & Data Streams:** OpenWeather API, AISStream WebSocket
* **Document Generation:** FPDF

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/vessel-digital-twin.git](https://github.com/yourusername/vessel-digital-twin.git)
   cd vessel-digital-twin

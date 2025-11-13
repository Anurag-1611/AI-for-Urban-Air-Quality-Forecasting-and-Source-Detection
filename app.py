# app_rf.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

# ---------- Page config ----------
st.set_page_config(
    page_title="Green Air — Urban Air Quality Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS: green theme, cards, header ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background: linear-gradient(180deg, #f3fff6 0%, #e8fff0 100%);
        color: #013220;
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        gap: 16px;
        background: linear-gradient(90deg,#7dd3a6,#16a34a);
        padding: 18px 22px;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(6,95,70,0.12);
        color: white;
    }
    .app-title { font-size: 22px; font-weight: 700; margin: 0; }
    .app-sub { font-size: 12.5px; opacity: 0.95; margin: 0; }

    /* Card containers */
    .card {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.05);
        margin-bottom: 14px;
    }

    .muted { color: #3a6350; font-size: 13px; }
    .small { font-size: 13px; color: #2b6f50; }

    /* Animated green pulse for result */
    .pulse {
        display:inline-block;
        padding:8px 14px;
        border-radius: 999px;
        background: linear-gradient(90deg,#bbf7d0,#86efac);
        color:#064e3b;
        font-weight:600;
        box-shadow: 0 6px 18px rgba(16,185,129,0.12);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-4px); }
        100% { transform: translateY(0px); }
    }

    /* Input labels smaller */
    .stSlider > label, .stSelectbox > label { color: #0b6b45; font-weight:600; }

    /* Narrow preview table style */
    .input-preview td { padding:6px 8px; font-size:13px; color:#024731; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Small SVG leaf for header ----------
leaf_svg = """
<svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M2 12C2 12 6 6 12 6C14 6 16 6 20 8C20 8 14 14 10 18C6 22 2 12 2 12Z" fill="#fff"/>
  <path d="M14 4C12.5 6 12 8 12 8" stroke="#9AE6B4" stroke-width="1.2" stroke-linecap="round"/>
</svg>
"""

# ---------- Load models ----------
@st.cache_resource
def load_models():
    model_dir = 'models'
    result = {'pollution_model': None, 'source_model': None, 'source_le': None, 'errors': []}

    def try_load(path):
        if not os.path.exists(path):
            return None, f"NOT FOUND: {path}"
        try:
            obj = joblib.load(path)
            return obj, None
        except Exception as e:
            try:
                import pickle
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                return obj, None
            except Exception as e2:
                return None, f"LOAD ERROR {path}: joblib:{e} pickle:{e2}"

    pm_path = os.path.join('models','rf_pollution_likely.pkl')
    src_path = os.path.join('models','rf_source.pkl')
    le_path = os.path.join('models','label_encoder_source.pkl')

    obj, err = try_load(pm_path); result['pollution_model'] = obj
    if err: result['errors'].append(err)
    obj, err = try_load(src_path); result['source_model'] = obj
    if err: result['errors'].append(err)
    obj, err = try_load(le_path); result['source_le'] = obj
    if err: result['errors'].append(err)
    return result

models = load_models()

# ---------- Header display ----------
header_html = f"""
<div class="app-header">
  <div>{leaf_svg}</div>
  <div>
    <div class="app-title">Green Air — Urban Air Quality Forecast</div>
    <div class="app-sub">Early warning & cause detection • Sustainable city planning assistant</div>
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

# ---------- Show model load status ----------
status_cols = st.columns(3)
with status_cols[0]:
    if models['pollution_model'] is not None:
        st.success("Pollution model ✅")
    else:
        st.warning("Pollution model missing")
with status_cols[1]:
    if models['source_model'] is not None:
        st.success("Source model ✅")
    else:
        st.info("Source model missing")
with status_cols[2]:
    if models['source_le'] is not None:
        st.success("Label encoder ✅")
    else:
        st.info("Label encoder missing")
if models['errors']:
    with st.expander("Model load messages"):
        for e in models['errors']:
            st.write("- " + e)

st.write("---")

# ---------- Left: Inputs (tabs/nav) ----------
left, right = st.columns([1, 1.4], gap="large")

with left:
    st.markdown('<div class="card"><strong>Inputs</strong> <div class="muted">Provide current sensor/observational values</div></div>', unsafe_allow_html=True)
    tabs = st.tabs(["Air Quality", "Meteorological Aspects", "Traffic Details"])

    # Air Quality tab
    with tabs[0]:
        st.markdown('<div class="card"><strong>Air Quality</strong></div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            pm25 = st.slider('PM2.5 (µg/m³)', 0.0, 500.0, 75.0, 0.1)
            pm10 = st.slider('PM10 (µg/m³)', 0.0, 500.0, 120.0, 0.1)
            no2 = st.slider('NO₂ (µg/m³)', 0.0, 300.0, 40.0, 0.1)
        with a2:
            so2 = st.slider('SO₂ (µg/m³)', 0.0, 200.0, 10.0, 0.1)
            co = st.slider('CO (ppm)', 0.0, 20.0, 0.8, 0.01)
            o3 = st.slider('O₃ (µg/m³)', 0.0, 300.0, 30.0, 0.1)

    # Meteorological tab
    with tabs[1]:
        st.markdown('<div class="card"><strong>Meteorological Aspects</strong></div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            temperature = st.slider('Temperature (°C)', -20.0, 50.0, 25.0, 0.1)
            humidity = st.slider('Humidity (%)', 0.0, 100.0, 60.0, 0.1)
            wind_speed = st.slider('Wind Speed (m/s)', 0.0, 40.0, 3.0, 0.1)
        with m2:
            wind_dir_choice = st.selectbox('Wind Direction', ['N','NE','E','SE','S','SW','W','NW'])
            rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 0.0, 0.1)
            pressure = st.slider('Pressure (hPa)', 800.0, 1100.0, 1010.0, 0.1)

    # Traffic tab
    with tabs[2]:
        st.markdown('<div class="card"><strong>Traffic Details</strong></div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            vehicle_count = st.number_input('Vehicle Count', min_value=0, max_value=200000, value=1000, step=1)
            avg_speed = st.slider('Average Speed (km/h)', 0.0, 200.0, 40.0, 0.1)
        with t2:
            congestion = st.slider('Congestion Level (0.0 - 1.0)', 0.0, 1.0, 0.3, 0.01)
            road_density = st.slider('Road Density (km/km²)', 0.0, 200.0, 8.0, 0.1)

    # numeric encoding for wind direction (match training)
    wind_map = {'N':0,'NE':1,'E':2,'SE':3,'S':4,'SW':5,'W':6,'NW':7}
    wind_dir = wind_map.get(wind_dir_choice, 0)

    # build input dataframe
    input_data = {
        'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2, 'CO': co, 'O3': o3,
        'Temperature': temperature, 'Humidity': humidity, 'Wind_Speed': wind_speed, 'Wind_Direction': wind_dir,
        'Rainfall': rainfall, 'Pressure': pressure, 'Vehicle_Count': vehicle_count,
        'Average_Speed': avg_speed, 'Congestion_Level': congestion, 'Road_Density': road_density
    }
    input_df = pd.DataFrame(input_data, index=[0])

    with st.expander("Preview input values"):
        preview = input_df.T.rename(columns={0:'Value'})
        preview_html = preview.to_html(classes="input-preview")
        st.write(preview_html, unsafe_allow_html=True)

# ---------- Right: results, explanation ----------
with right:
    st.markdown('<div class="card"><strong>Prediction & Explanation</strong></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict (Green Air)")

    # helper to align features
    def align_features_for_model(X: pd.DataFrame, model):
        Xc = X.copy()
        if model is not None and hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
            for col in expected:
                if col not in Xc.columns:
                    Xc[col] = 0.0
            Xc = Xc[expected]
        Xc = Xc.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return Xc

    # simple rule-based explanation
    def rule_based_reason(row):
        reasons = []
        if row['Vehicle_Count'] > 3000 or row['NO2'] > 80 or row['CO'] > 2.5:
            reasons.append(('Vehicular Emission', [
                f"Vehicle_Count = {int(row['Vehicle_Count'])}" if row['Vehicle_Count']>3000 else None,
                f"NO2 = {row['NO2']:.1f} (high)" if row['NO2']>80 else None,
                f"CO = {row['CO']:.2f} (high)" if row['CO']>2.5 else None
            ]))
        if row['SO2'] > 50 or row['CO'] > 3.5:
            reasons.append(('Industrial Emission', [
                f"SO2 = {row['SO2']:.1f} (high)" if row['SO2']>50 else None,
                f"CO = {row['CO']:.2f} (very high)" if row['CO']>3.5 else None
            ]))
        if row['PM2.5'] > 120 and row['Rainfall'] < 5 and row['Wind_Speed'] < 2:
            reasons.append(('Dust / Poor Dispersion', [
                f"PM2.5 = {row['PM2.5']:.1f} (very high)",
                f"Rainfall = {row['Rainfall']:.1f} (low)",
                f"Wind_Speed = {row['Wind_Speed']:.1f} (low)"
            ]))
        if not reasons:
            reasons.append(('Low Pollution Source', [
                "No major pollutant exceeded heuristic thresholds",
                f"PM2.5 = {row['PM2.5']:.1f}, NO2 = {row['NO2']:.1f}, SO2 = {row['SO2']:.1f}"
            ]))
        return reasons

    # run prediction
    if predict_btn:
        try:
            poll_model = models['pollution_model']
            src_model = models['source_model']
            le = models['source_le']

            if poll_model is None:
                st.error("Pollution model not found. Place rf_pollution_likely.pkl in models/ and restart.")
            else:
                Xp = align_features_for_model(input_df, poll_model)
                pred_raw = poll_model.predict(Xp)[0]
                try:
                    poll_yesno = 'Yes' if int(pred_raw) == 1 else 'No'
                except Exception:
                    poll_yesno = str(pred_raw)

                # nice result badge
                st.markdown("<div style='display:flex;gap:12px;align-items:center'>", unsafe_allow_html=True)
                if poll_yesno == 'Yes':
                    st.markdown('<div class="pulse">Pollution Likely: YES</div>', unsafe_allow_html=True)
                else:
                    st.success("Pollution Likely: NO")
                st.markdown("</div>", unsafe_allow_html=True)

                # probabilities
                try:
                    if hasattr(poll_model, 'predict_proba'):
                        probs = poll_model.predict_proba(Xp)[0]
                        proba_df = pd.DataFrame({'Class': poll_model.classes_, 'Probability': probs}).set_index('Class')
                        st.subheader("Prediction Probabilities")
                        st.dataframe(proba_df, use_container_width=True)
                except Exception:
                    pass

                # decode main source
                decoded_label = None
                decode_notes = []
                if poll_yesno == 'Yes' and src_model is not None:
                    try:
                        Xs = align_features_for_model(input_df, src_model)
                        src_raw = src_model.predict(Xs)[0]

                        # try label encoder
                        if le is not None:
                            try:
                                if isinstance(src_raw, (int, np.integer)):
                                    decoded_label = le.inverse_transform([int(src_raw)])[0]
                                else:
                                    decoded_label = le.inverse_transform([str(src_raw)])[0]
                            except Exception as e:
                                decode_notes.append(f"LabelEncoder inverse failed: {e}")
                                decoded_label = None

                        # try src_model.classes_
                        if decoded_label is None and hasattr(src_model, 'classes_'):
                            try:
                                classes = list(src_model.classes_)
                                if isinstance(src_raw, (int, np.integer)):
                                    idx = int(src_raw)
                                    if 0 <= idx < len(classes):
                                        decoded_label = classes[idx]
                                    else:
                                        decoded_label = str(src_raw)
                                else:
                                    decoded_label = str(src_raw)
                            except Exception as e:
                                decode_notes.append(f"classes_ decode failed: {e}")
                                decoded_label = None

                        # fallback mapping
                        if decoded_label is None:
                            fallback = {0:'Vehicular Emission', 1:'Industrial Emission', 2:'Dust / Poor Dispersion', 3:'Low Pollution Source'}
                            try:
                                decoded_label = fallback.get(int(src_raw), str(src_raw))
                            except Exception:
                                decoded_label = str(src_raw)

                        st.subheader("Predicted Main Source")
                        st.info(f"**{decoded_label}**")

                        if decode_notes:
                            with st.expander("Decoding notes"):
                                for n in decode_notes:
                                    st.write("- " + n)

                    except Exception as e:
                        st.warning("Source model prediction failed: " + str(e))
                        st.write(traceback.format_exc())
                else:
                    st.info("Main source prediction skipped (pollution not likely or source model missing).")

                # Explanation: rule-based reason
                st.subheader("Why (Dominant Factors)")
                reasons = rule_based_reason(input_df.iloc[0])
                chosen_reason = None
                if decoded_label is not None:
                    for rlabel, ev in reasons:
                        if rlabel.lower().split('/')[0].strip() in decoded_label.lower():
                            chosen_reason = (rlabel, ev)
                            break
                if chosen_reason is None:
                    chosen_reason = reasons[0]

                rlabel, evidence = chosen_reason
                st.write(f"**Most likely cause:** {rlabel}")
                st.markdown("**Evidence from inputs:**")
                for ev in evidence:
                    if ev:
                        st.write("- " + ev)

                # friendly summary
                summary = f"Model predicted **{decoded_label or 'N/A'}**; dominant factor appears to be **{rlabel}**."
                st.markdown(f"**Summary:** {summary}")

        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.write(traceback.format_exc())

st.write("---")
st.markdown("""
**Notes & next steps**
- Keep `models/` folder with `rf_pollution_likely.pkl`, `rf_source.pkl`, and `label_encoder_source.pkl` in the same folder as `app_rf.py`.
- For reliable production usage, bundle preprocessing + model into a single sklearn `Pipeline`.
- Want SHAP explanations (feature-level attribution)? I can add them — they'll make the app highly persuasive for reports/demos.
""")

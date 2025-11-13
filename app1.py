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
    initial_sidebar_state="collapsed"
)

# ---------- CSS: center title, visible tabs, clean green theme ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* App background & font */
    .stApp {
        background: linear-gradient(180deg, #f7fffa 0%, #f0fff4 100%);
        font-family: 'Inter', sans-serif;
        color: #023927;
    }

    /* Centered large title */
    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: 800;
        color: #064e3b;
        margin-bottom: 4px;
    }
    .main-sub {
        text-align: center;
        font-size: 14px;
        color: #096b43;
        margin-top: 0;
        margin-bottom: 18px;
    }

    /* Container that centers content on page */
    .center-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }

    /* Card style for input & output sections */
    .card {
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(4, 78, 40, 0.06);
        max-width: 900px;
        margin: 10px auto;
    }

    /* Make the tabs labels clearly visible and centered */
    /* Works for new Streamlit versions */
    .css-1f0j8pp { /* wrapper for tabs - class name may vary; this rule improves contrast */
        color: #064e3b !important;
    }
    /* Force tab button text color and background */
    button[role="tab"] > div {
        color: #064e3b !important;
        font-weight: 600;
    }
    /* Make active tab more visible */
    button[role="tab"][data-selected="true"] > div {
        background: linear-gradient(90deg,#a7f3d0,#34d399) !important;
        color: #063a2f !important;
        border-radius: 8px;
        padding: 6px 10px;
    }

    /* Center the predict button and make it prominent */
    .stButton>button {
        background: linear-gradient(90deg,#34d399,#10b981);
        color: white;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 10px;
        border: none;
    }

    /* Make small labels more readable */
    .stSlider > label, .stSelectbox > label {
        color: #064e3b;
        font-weight: 600;
    }

    /* Input preview table style */
    .input-preview td { padding: 6px 8px; color: #064e3b; }

    /* Output card text */
    .output-title { font-size: 18px; font-weight:700; color:#064e3b; }
    .output-sub { color:#0b6b45; font-size:14px; }

    /* Reduce sidebar width since we collapsed it */
    .css-1d391kg { width: 0px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helper: load models ----------
@st.cache_resource
def load_models():
    model_dir = 'models'
    out = {'pollution_model': None, 'source_model': None, 'source_le': None, 'errors': []}

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

    pm_path = os.path.join(model_dir, 'rf_pollution_likely.pkl')
    src_path = os.path.join(model_dir, 'rf_source.pkl')
    le_path = os.path.join(model_dir, 'label_encoder_source.pkl')

    obj, err = try_load(pm_path); out['pollution_model'] = obj
    if err: out['errors'].append(err)
    obj, err = try_load(src_path); out['source_model'] = obj
    if err: out['errors'].append(err)
    obj, err = try_load(le_path); out['source_le'] = obj
    if err: out['errors'].append(err)

    return out

models = load_models()

# ---------- Page header (centered) ----------
st.markdown('<div class="main-title">🌿 Green Air — Urban Air Quality Forecast and Source Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Early warning & cause detection for healthier cities</div>', unsafe_allow_html=True)

# ---------- Centered tabbed inputs below title ----------
# We use a centered card that contains the tabs and the inputs inside.
st.markdown('<div class="card">', unsafe_allow_html=True)

# Create centered container using columns
coll, colc, colr = st.columns([1, 2, 1])
with colc:
    # create tabs (nav)
    tabs = st.tabs(["Air Quality", "Meteorological Aspects", "Traffic Details"])

    # --- Air Quality tab ---
    with tabs[0]:
        st.markdown("### Air Quality")
        a1, a2 = st.columns(2)
        with a1:
            pm25 = st.slider('PM2.5 (µg/m³)', 0.0, 500.0, 75.0, 0.1)
            pm10 = st.slider('PM10 (µg/m³)', 0.0, 500.0, 120.0, 0.1)
            no2 = st.slider('NO₂ (µg/m³)', 0.0, 300.0, 40.0, 0.1)
        with a2:
            so2 = st.slider('SO₂ (µg/m³)', 0.0, 200.0, 10.0, 0.1)
            co = st.slider('CO (ppm)', 0.0, 20.0, 0.8, 0.01)
            o3 = st.slider('O₃ (µg/m³)', 0.0, 300.0, 30.0, 0.1)

    # --- Meteorological tab ---
    with tabs[1]:
        st.markdown("### Meteorological Aspects")
        m1, m2 = st.columns(2)
        with m1:
            temperature = st.slider('Temperature (°C)', -20.0, 50.0, 25.0, 0.1)
            humidity = st.slider('Humidity (%)', 0.0, 100.0, 60.0, 0.1)
            wind_speed = st.slider('Wind Speed (m/s)', 0.0, 40.0, 3.0, 0.1)
        with m2:
            wind_dir_choice = st.selectbox('Wind Direction', ['N','NE','E','SE','S','SW','W','NW'])
            rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 0.0, 0.1)
            pressure = st.slider('Pressure (hPa)', 800.0, 1100.0, 1010.0, 0.1)

    # --- Traffic tab ---
    with tabs[2]:
        st.markdown("### Traffic Details")
        t1, t2 = st.columns(2)
        with t1:
            vehicle_count = st.number_input('Vehicle Count', min_value=0, max_value=200000, value=1000, step=1)
            avg_speed = st.slider('Average Speed (km/h)', 0.0, 200.0, 40.0, 0.1)
        with t2:
            congestion = st.slider('Congestion Level (0.0 - 1.0)', 0.0, 1.0, 0.3, 0.01)
            road_density = st.slider('Road Density (km/km²)', 0.0, 200.0, 8.0, 0.1)

    # wind direction numeric mapping (must match training)
    wind_map = {'N':0,'NE':1,'E':2,'SE':3,'S':4,'SW':5,'W':6,'NW':7}
    wind_dir = wind_map.get(wind_dir_choice, 0)

    # Pack inputs into one-row DataFrame
    input_data = {
        'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2, 'CO': co, 'O3': o3,
        'Temperature': temperature, 'Humidity': humidity, 'Wind_Speed': wind_speed, 'Wind_Direction': wind_dir,
        'Rainfall': rainfall, 'Pressure': pressure, 'Vehicle_Count': vehicle_count,
        'Average_Speed': avg_speed, 'Congestion_Level': congestion, 'Road_Density': road_density
    }
    input_df = pd.DataFrame(input_data, index=[0])

    # show compact preview
    with st.expander("Preview input values"):
        st.table(input_df.T.rename(columns={0:'Value'}))

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Predict button centered below inputs ----------
btn_col_left, btn_col_center, btn_col_right = st.columns([1, 0.4, 1])
with btn_col_center:
    predict_clicked = st.button("🔮 Predict", help="Click to predict pollution likelihood and main source")

# ---------- Output area: centered card below button ----------
out_left, out_center, out_right = st.columns([1, 2, 1])
with out_center:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # output content will be placed here after click
    if predict_clicked:
        # helper to align features to model
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

        # rule-based reason generator
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

        # Run prediction
        try:
            poll_model = models['pollution_model']
            src_model = models['source_model']
            le = models['source_le']

            if poll_model is None:
                st.error("Pollution model not loaded. Place rf_pollution_likely.pkl in models/ and restart app.")
            else:
                Xp = align_features_for_model(input_df, poll_model)
                pred_raw = poll_model.predict(Xp)[0]
                try:
                    poll_yesno = 'Yes' if int(pred_raw) == 1 else 'No'
                except Exception:
                    poll_yesno = str(pred_raw)

                # Display pollution result prominently
                st.markdown('<div class="output-title">Pollution Prediction</div>', unsafe_allow_html=True)
                if poll_yesno == 'Yes':
                    st.markdown('<div style="color:#b91c1c;font-weight:800;font-size:20px">⚠️ Pollution Likely: YES</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#065f46;font-weight:800;font-size:20px">✅ Pollution Likely: NO</div>', unsafe_allow_html=True)

                # Probability table (if available)
                try:
                    if hasattr(poll_model, 'predict_proba'):
                        probs = poll_model.predict_proba(Xp)[0]
                        proba_df = pd.DataFrame({'Class': poll_model.classes_, 'Probability': probs}).set_index('Class')
                        st.subheader("Prediction Probabilities")
                        st.dataframe(proba_df, use_container_width=True)
                except Exception:
                    pass

                # Predict main source and decode
                decoded_label = None
                decode_notes = []
                if poll_yesno == 'Yes' and src_model is not None:
                    try:
                        Xs = align_features_for_model(input_df, src_model)
                        src_raw = src_model.predict(Xs)[0]

                        # Attempt decoding using saved LabelEncoder
                        if le is not None:
                            try:
                                if isinstance(src_raw, (int, np.integer)):
                                    decoded_label = le.inverse_transform([int(src_raw)])[0]
                                else:
                                    decoded_label = le.inverse_transform([str(src_raw)])[0]
                            except Exception as e:
                                decode_notes.append(f"LabelEncoder inverse failed: {e}")
                                decoded_label = None

                        # If still None, try src_model.classes_
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

                        # final fallback mapping
                        if decoded_label is None:
                            fallback = {0:'Vehicular Emission', 1:'Industrial Emission', 2:'Dust / Poor Dispersion', 3:'Low Pollution Source'}
                            try:
                                decoded_label = fallback.get(int(src_raw), str(src_raw))
                            except Exception:
                                decoded_label = str(src_raw)

                        st.subheader("Predicted Main Source")
                        st.markdown(f"<div class='output-sub'><strong>{decoded_label}</strong></div>", unsafe_allow_html=True)

                        if decode_notes:
                            with st.expander("Decoding notes"):
                                for n in decode_notes:
                                    st.write("- " + n)

                    except Exception as e:
                        st.warning("Source prediction failed: " + str(e))
                        st.write(traceback.format_exc())
                else:
                    st.info("Main source prediction skipped (pollution not likely or source model missing).")

                # Rule-based explanation and evidence
                st.subheader("Why (Dominant Factors / Explanation)")
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
                st.markdown("**Evidence from current inputs:**")
                for ev in evidence:
                    if ev:
                        st.write("- " + ev)

                st.markdown(f"**Summary:** Model predicted **{decoded_label or 'N/A'}**; dominant factor appears to be **{rlabel}**.")

        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.write(traceback.format_exc())

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.write("")
st.markdown("<div style='text-align:center;color:#024731;font-size:12px'>Tip: keep the models/ folder (rf_pollution_likely.pkl, rf_source.pkl, label_encoder_source.pkl) in the same directory as this app.</div>", unsafe_allow_html=True)

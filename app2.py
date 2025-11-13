# app_rf.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

st.set_page_config(page_title="Green Air", page_icon="🌿", layout="wide")

# ---------- CSS: dark rounded navbar pill + layout ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background: linear-gradient(180deg,#f6fff7 0%,#f0fff4 100%); color:#083826; }

    /* Main title centered */
    .main-title { text-align:center; font-size:34px; font-weight:800; color:#064e3b; margin-top:12px; margin-bottom:4px; }
    .sub-title { text-align:center; color:#0b6b45; margin-bottom:18px; }

    /* Navbar pill: centered, dark, rounded with shadow */
    .nav-pill {
      width:100%;
      max-width:980px;
      margin: 0 auto 18px auto;
      background: linear-gradient(90deg,#111111,#222222);
      padding: 10px 18px;
      border-radius: 999px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.25);
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
    }

    .nav-left { display:flex; align-items:center; gap:12px; }
    .brand {
      display:flex; align-items:center; gap:10px;
      color:#fff; font-weight:800; font-size:16px;
    }
    .nav-links { display:flex; align-items:center; gap:8px; justify-content:center; flex:1; }

    /* pill-like nav items */
    .nav-item {
      color:#dfeee7;
      padding:10px 16px;
      border-radius:999px;
      font-weight:700;
      cursor:pointer;
      background: transparent;
      border: 1px solid rgba(255,255,255,0.04);
      transition: all 150ms ease;
    }
    .nav-item:hover { transform: translateY(-2px); background: rgba(255,255,255,0.03); }
    .nav-item-active {
      background: white;
      color: #b91c1c;
      box-shadow: 0 8px 18px rgba(0,0,0,0.12);
      transform: translateY(-2px);
    }

    /* right-most pill (like email pill) */
    .nav-pill-right {
      background: white;
      color: #0b3e2c;
      padding:8px 14px;
      border-radius:999px;
      font-weight:700;
      box-shadow: 0 8px 18px rgba(0,0,0,0.12);
    }

    /* content card */
    .content-card { background:white; border-radius:12px; padding:20px; max-width:980px; margin: 0 auto 18px auto; box-shadow: 0 10px 30px rgba(4,78,40,0.04); }

    /* predict button */
    .stButton>button {
      background: linear-gradient(90deg,#10b981,#047857) !important;
      color:white !important;
      font-weight:800;
      padding:12px 30px;
      border-radius:12px;
      font-size:18px;
    }

    /* output card */
    .output-card { background:#f7fffb; border-radius:12px; padding:18px; max-width:980px; margin: 12px auto 30px auto; box-shadow: 0 10px 24px rgba(4,78,40,0.04); }

    /* most likely cause highlight */
    .most-likely { display:inline-block; background: linear-gradient(90deg,#fff7d6,#ffd88a); padding:12px 18px; border-radius:10px; font-weight:800; color:#1f2937; font-size:18px; box-shadow: 0 8px 18px rgba(0,0,0,0.06); }

    label { color:#064e3b !important; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Load models helper ----------
@st.cache_resource
def load_models():
    model_dir = 'models'
    out = {'pollution_model': None, 'source_model': None, 'source_le': None, 'errors': []}

    def try_load(path):
        if not os.path.exists(path):
            return None, f"NOT FOUND: {path}"
        try:
            return joblib.load(path), None
        except Exception as e:
            try:
                import pickle
                with open(path, 'rb') as f:
                    return pickle.load(f), None
            except Exception as e2:
                return None, f"LOAD ERROR {path}: joblib:{e} pickle:{e2}"

    paths = {
        'pollution_model': os.path.join(model_dir, 'rf_pollution_likely.pkl'),
        'source_model': os.path.join(model_dir, 'rf_source.pkl'),
        'source_le': os.path.join(model_dir, 'label_encoder_source.pkl')
    }
    for k, p in paths.items():
        obj, err = try_load(p)
        out[k] = obj
        if err: out['errors'].append(err)
    return out

models = load_models()

# ---------- Page header ----------
st.markdown('<div class="main-title">🌿 Green Air — Urban Air Quality Forecast & Source Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Early warning and cause detection for healthier cities</div>', unsafe_allow_html=True)

# ---------- Navbar pill: render and use session_state for selected section ----------
if 'selected_section' not in st.session_state:
    st.session_state['selected_section'] = 'Air Quality'

# Create navbar layout
def render_navbar():
    # Outer container
    st.markdown('<div class="nav-pill">', unsafe_allow_html=True)
    # left brand
    st.markdown('<div class="nav-left"><div class="brand">🌱 <span style="margin-left:6px">Green Air</span></div></div>', unsafe_allow_html=True)

    # center links - we will create small buttons; clicking sets session_state
    nav_cols = st.columns([1,1,1,0.6])  # last for right pill
    labels = ["Air Quality", "Meteorological Aspects", "Traffic Details"]
    for i, label in enumerate(labels):
        key = f"nav_btn_{i}"
        # Render as normal Streamlit button; CSS will style it using the label text
        is_active = (st.session_state['selected_section'] == label)
        if is_active:
            # show visually active button (we still use a normal button but clicking re-selects)
            if st.button(label, key=key):
                st.session_state['selected_section'] = label
            # apply active style by injecting JS-less CSS: add a small hack by writing a span next to it.
            # Styling is handled by global CSS targeting button text; acceptable for most Streamlit versions.
        else:
            if st.button(label, key=key):
                st.session_state['selected_section'] = label

    # right small pill (status/email-like)
    with st.container():
        # simple text, not interactive
        status_text = []
        if models['pollution_model'] is not None:
            status_text.append("Model OK")
        if models['source_model'] is not None:
            status_text.append("Source OK")
        status = " • ".join(status_text) if status_text else "Models missing"
        st.markdown(f'<div class="nav-pill-right">{status}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

render_navbar()

# ---------- Content card (centered). Show only subsections for selected section ----------
st.markdown('<div class="content-card">', unsafe_allow_html=True)

st.markdown(f"### {st.session_state['selected_section']}", unsafe_allow_html=True)

# input variables (we keep them persisted by using keys)
# Only render controls for the selected section — others hidden until clicked
if st.session_state['selected_section'] == 'Air Quality':
    c1, c2 = st.columns(2)
    with c1:
        pm25 = st.slider('PM2.5 (µg/m³)', 0.0, 500.0, 75.0, 0.1, key='pm25')
        pm10 = st.slider('PM10 (µg/m³)', 0.0, 500.0, 120.0, 0.1, key='pm10')
        no2 = st.slider('NO₂ (µg/m³)', 0.0, 300.0, 40.0, 0.1, key='no2')
    with c2:
        so2 = st.slider('SO₂ (µg/m³)', 0.0, 200.0, 10.0, 0.1, key='so2')
        co = st.slider('CO (ppm)', 0.0, 20.0, 0.8, 0.01, key='co')
        o3 = st.slider('O₃ (µg/m³)', 0.0, 300.0, 30.0, 0.1, key='o3')

elif st.session_state['selected_section'] == 'Meteorological Aspects':
    c1, c2 = st.columns(2)
    with c1:
        temperature = st.slider('Temperature (°C)', -20.0, 50.0, 25.0, 0.1, key='temp')
        humidity = st.slider('Humidity (%)', 0.0, 100.0, 60.0, 0.1, key='hum')
        wind_speed = st.slider('Wind Speed (m/s)', 0.0, 40.0, 3.0, 0.1, key='ws')
    with c2:
        wind_dir_choice = st.selectbox('Wind Direction', ['N','NE','E','SE','S','SW','W','NW'], key='wd')
        rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 0.0, 0.1, key='rain')
        pressure = st.slider('Pressure (hPa)', 800.0, 1100.0, 1010.0, 0.1, key='pres')

elif st.session_state['selected_section'] == 'Traffic Details':
    c1, c2 = st.columns(2)
    with c1:
        vehicle_count = st.number_input('Vehicle Count', min_value=0, max_value=200000, value=1000, step=1, key='vc')
        avg_speed = st.slider('Average Speed (km/h)', 0.0, 200.0, 40.0, 0.1, key='aspeed')
    with c2:
        congestion = st.slider('Congestion Level (0.0 - 1.0)', 0.0, 1.0, 0.3, 0.01, key='cong')
        road_density = st.slider('Road Density (km/km²)', 0.0, 200.0, 8.0, 0.1, key='rd')

# We still want the final prediction to use the combined set of feature values.
# For features that might not have been displayed yet (because the user didn't click that section),
# we attempt to read them from session_state or fill defaults.
def get_val(key, default):
    return st.session_state.get(key, default)

# Read or default all features:
pm25 = get_val('pm25', 75.0)
pm10 = get_val('pm10', 120.0)
no2 = get_val('no2', 40.0)
so2 = get_val('so2', 10.0)
co = get_val('co', 0.8)
o3 = get_val('o3', 30.0)
temperature = get_val('temp', 25.0)
humidity = get_val('hum', 60.0)
wind_speed = get_val('ws', 3.0)
wind_dir_choice = get_val('wd', 'N')
rainfall = get_val('rain', 0.0)
pressure = get_val('pres', 1010.0)
vehicle_count = get_val('vc', 1000)
avg_speed = get_val('aspeed', 40.0)
congestion = get_val('cong', 0.3)
road_density = get_val('rd', 8.0)

# numeric wind encode
wind_map = {'N':0,'NE':1,'E':2,'SE':3,'S':4,'SW':5,'W':6,'NW':7}
wind_dir = wind_map.get(wind_dir_choice, 0)

# Build input dataframe
input_data = {
    'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2, 'CO': co, 'O3': o3,
    'Temperature': temperature, 'Humidity': humidity, 'Wind_Speed': wind_speed, 'Wind_Direction': wind_dir,
    'Rainfall': rainfall, 'Pressure': pressure, 'Vehicle_Count': vehicle_count,
    'Average_Speed': avg_speed, 'Congestion_Level': congestion, 'Road_Density': road_density
}
input_df = pd.DataFrame(input_data, index=[0])

# Preview small
with st.expander("Preview input values", expanded=False):
    st.table(input_df.T.rename(columns={0:'Value'}))

st.markdown('</div>', unsafe_allow_html=True)  # close content-card

# ---------- Predict button centered under content ----------
c1, c2, c3 = st.columns([1, 0.4, 1])
with c2:
    predict_clicked = st.button("🔮 Predict", key='predict')

# ---------- Output area ----------
if predict_clicked:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    st.markdown("## Result", unsafe_allow_html=True)

    # helpers
    def align_features(X: pd.DataFrame, model):
        Xc = X.copy()
        if model is not None and hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
            for col in expected:
                if col not in Xc.columns:
                    Xc[col] = 0.0
            Xc = Xc[expected]
        Xc = Xc.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return Xc

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
    try:
        poll_model = models['pollution_model']
        src_model = models['source_model']
        le = models['source_le']

        if poll_model is None:
            st.error("Pollution model not loaded. Put rf_pollution_likely.pkl in models/ and restart.")
        else:
            Xp = align_features(input_df, poll_model)
            pred_raw = poll_model.predict(Xp)[0]
            try:
                poll_yesno = 'Yes' if int(pred_raw) == 1 else 'No'
            except Exception:
                poll_yesno = str(pred_raw)

            # show pollution result
            if poll_yesno == 'Yes':
                st.markdown("<div style='font-weight:800;color:#b91c1c;font-size:20px'>⚠️ Pollution Likely: YES</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-weight:800;color:#065f46;font-size:20px'>✅ Pollution Likely: NO</div>", unsafe_allow_html=True)

            # probs
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
                    Xs = align_features(input_df, src_model)
                    src_raw = src_model.predict(Xs)[0]

                    # try LabelEncoder
                    if le is not None:
                        try:
                            if isinstance(src_raw, (int, np.integer)):
                                decoded_label = le.inverse_transform([int(src_raw)])[0]
                            else:
                                decoded_label = le.inverse_transform([str(src_raw)])[0]
                        except Exception as e:
                            decode_notes.append(f"LabelEncoder inverse failed: {e}")
                            decoded_label = None

                    # fallback to classes_
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
                    st.markdown(f"<div style='font-size:16px;font-weight:700;color:#034d3a'>{decoded_label}</div>", unsafe_allow_html=True)

                    if decode_notes:
                        with st.expander("Decoding notes"):
                            for n in decode_notes:
                                st.write("- " + n)
                except Exception as e:
                    st.warning("Source model prediction failed: " + str(e))
                    st.write(traceback.format_exc())
            else:
                st.info("Main source prediction skipped (pollution not likely or source model missing).")

            # explanation
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
            st.markdown(f"<div class='most-likely'>Most likely cause: {rlabel}</div>", unsafe_allow_html=True)
            st.markdown("**Evidence from current inputs:**")
            for ev in evidence:
                if ev:
                    st.write("- " + ev)

            st.markdown(f"**Summary:** Model predicted **{decoded_label or 'N/A'}**; dominant factor appears to be **{rlabel}**.")

    except Exception as e:
        st.error("Prediction failed: " + str(e))
        st.write(traceback.format_exc())

    st.markdown('</div>', unsafe_allow_html=True)  # close output-card

# ---------- Footer ----------
st.markdown("<div style='text-align:center;color:#064e3b;font-size:12px;margin-top:12px'>Keep the models/ folder (rf_pollution_likely.pkl, rf_source.pkl, label_encoder_source.pkl) next to this app.</div>", unsafe_allow_html=True)

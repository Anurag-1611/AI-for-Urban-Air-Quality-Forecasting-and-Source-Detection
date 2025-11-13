# app_rf.py

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import traceback

# --- Page config ---
st.set_page_config(page_title="Urban Air Quality Predictor", page_icon="🌫️", layout="wide")

# --- Load models & encoder ---
@st.cache_resource
def load_models():
    model_dir = 'models'
    models = {'pollution_model': None, 'source_model': None, 'source_le': None, 'errors': []}

    # load helpers
    def try_joblib_load(path):
        if not os.path.exists(path):
            return None, f"NOT FOUND: {path}"
        try:
            obj = joblib.load(path)
            return obj, None
        except Exception as e:
            # try pickle fallback
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

    obj, err = try_joblib_load(pm_path)
    models['pollution_model'] = obj
    if err: models['errors'].append(err)

    obj, err = try_joblib_load(src_path)
    models['source_model'] = obj
    if err: models['errors'].append(err)

    obj, err = try_joblib_load(le_path)
    models['source_le'] = obj
    if err: models['errors'].append(err)

    return models

models = load_models()

# status UI
if models['pollution_model'] is None:
    st.warning("Pollution model (rf_pollution_likely.pkl) NOT loaded. Place it in models/ folder.")
else:
    st.success("Pollution model loaded.")

if models['source_model'] is None:
    st.info("Source model (rf_source.pkl) not loaded — source prediction will be skipped.")
else:
    st.success("Source model loaded.")

if models['source_le'] is None:
    st.info("Label encoder (label_encoder_source.pkl) not loaded — decoding may fall back to model.classes_ or heuristics.")
else:
    st.success("Label encoder loaded.")

if models['errors']:
    with st.expander("Model load messages"):
        st.write("\n".join(models['errors']))

# --- Title & description ---
st.title("🌫️ Urban Air Quality Forecasting & Cause Explanation")
st.markdown("""
Predicts whether pollution is likely and shows the predicted main source (Vehicular / Industrial / Dust / Low Pollution),  
plus a short human-readable explanation of the dominating factor(s) for the current inputs.
""")
st.write("---")

# --- Sidebar inputs ---
st.sidebar.header("Input Features")

def user_input_features():
    pm25 = st.sidebar.slider('PM2.5 (µg/m³)', 0.0, 500.0, 75.0)
    pm10 = st.sidebar.slider('PM10 (µg/m³)', 0.0, 500.0, 120.0)
    no2 = st.sidebar.slider('NO₂ (µg/m³)', 0.0, 300.0, 40.0)
    so2 = st.sidebar.slider('SO₂ (µg/m³)', 0.0, 200.0, 10.0)
    co = st.sidebar.slider('CO (ppm)', 0.0, 20.0, 0.8)
    o3 = st.sidebar.slider('O₃ (µg/m³)', 0.0, 300.0, 30.0)

    temperature = st.sidebar.slider('Temperature (°C)', -10.0, 50.0, 25.0)
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 60.0)
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', 0.0, 30.0, 3.0)
    wind_dir_choice = st.sidebar.selectbox('Wind Direction', ['N','NE','E','SE','S','SW','W','NW'])
    rainfall = st.sidebar.slider('Rainfall (mm)', 0.0, 200.0, 0.0)
    pressure = st.sidebar.slider('Pressure (hPa)', 800.0, 1100.0, 1010.0)

    vehicle_count = st.sidebar.number_input('Vehicle Count', min_value=0, max_value=1000000, value=1000, step=1)
    avg_speed = st.sidebar.slider('Average Speed (km/h)', 0.0, 200.0, 40.0)
    congestion = st.sidebar.slider('Congestion Level (0.0 - 1.0)', 0.0, 1.0, 0.3)
    road_density = st.sidebar.slider('Road Density (km/km²)', 0.0, 100.0, 8.0)

    wind_mapping = {'N':0,'NE':1,'E':2,'SE':3,'S':4,'SW':5,'W':6,'NW':7}
    wind_dir = wind_mapping[wind_dir_choice]

    data = {
        'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2, 'CO': co, 'O3': o3,
        'Temperature': temperature, 'Humidity': humidity, 'Wind_Speed': wind_speed, 'Wind_Direction': wind_dir,
        'Rainfall': rainfall, 'Pressure': pressure,
        'Vehicle_Count': vehicle_count, 'Average_Speed': avg_speed, 'Congestion_Level': congestion, 'Road_Density': road_density
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("Input Parameters")
st.dataframe(input_df, use_container_width=True)
st.write("---")

# --- Alignment helper ---
def align_features(X: pd.DataFrame, model):
    Xc = X.copy()
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in Xc.columns:
                Xc[col] = 0.0
        Xc = Xc[expected]
    # numeric coercion
    Xc = Xc.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return Xc

# --- Explain rules (simple domain heuristics) ---
def rule_based_reason(row):
    reasons = []
    # thresholds (you can tune)
    if row['Vehicle_Count'] > 3000 or row['NO2'] > 80 or row['CO'] > 2.5:
        reasons.append(('Vehicular Emission', [
            f"Vehicle_Count = {int(row['Vehicle_Count'])} (high)" if row['Vehicle_Count']>3000 else None,
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
    # If none matched, indicate low pollution contributors
    if not reasons:
        reasons.append(('Low Pollution Source', [
            "No major pollutant exceeded heuristic thresholds",
            f"PM2.5 = {row['PM2.5']:.1f}, NO2 = {row['NO2']:.1f}, SO2 = {row['SO2']:.1f}"
        ]))
    return reasons

# --- Predict button ---
if st.button("Predict"):
    try:
        pollution_model = models['pollution_model']
        source_model = models['source_model']
        le = models['source_le']

        if pollution_model is None:
            st.error("Pollution model missing. Place rf_pollution_likely.pkl in models/ and restart.")
        else:
            # Align input for pollution model
            Xp = align_features(input_df, pollution_model)
            raw_pred = pollution_model.predict(Xp)[0]
            # handle string labels or numeric
            try:
                poll_label = 'Yes' if int(raw_pred) == 1 else 'No'
            except Exception:
                poll_label = str(raw_pred)

            st.header("Prediction Result")
            if poll_label == 'Yes':
                st.error("⚠️ Pollution Likely: YES")
            else:
                st.success("✅ Pollution Likely: NO")

            # show probabilities if available
            if hasattr(pollution_model, 'predict_proba'):
                try:
                    probs = pollution_model.predict_proba(Xp)[0]
                    proba_df = pd.DataFrame({'Class': pollution_model.classes_, 'Probability': probs}).set_index('Class')
                    st.subheader("Prediction Probabilities")
                    st.dataframe(proba_df, use_container_width=True)
                except Exception:
                    pass

            # --- MAIN SOURCE DECODING (model + label encoder) ---
            decoded_label = None
            decode_errors = []

            if poll_label == 'Yes' and source_model is not None:
                try:
                    Xs = align_features(input_df, source_model)
                    src_raw = source_model.predict(Xs)[0]

                    # 1) Try LabelEncoder (preferred)
                    if le is not None:
                        try:
                            # sometimes src_raw is int index, sometimes str label
                            if isinstance(src_raw, (int, np.integer)):
                                decoded_label = le.inverse_transform([int(src_raw)])[0]
                            else:
                                decoded_label = le.inverse_transform([str(src_raw)])[0]
                        except Exception as e:
                            decode_errors.append(f"LabelEncoder inverse failed: {e}")
                            decoded_label = None

                    # 2) If encoder not available or failed, try model.classes_
                    if decoded_label is None and hasattr(source_model, 'classes_'):
                        try:
                            classes = list(source_model.classes_)
                            if isinstance(src_raw, (int, np.integer)):
                                idx = int(src_raw)
                                if 0 <= idx < len(classes):
                                    decoded_label = classes[idx]
                                else:
                                    decoded_label = str(src_raw)
                            else:
                                decoded_label = str(src_raw)
                        except Exception as e:
                            decode_errors.append(f"classes_ decode failed: {e}")
                            decoded_label = None

                    # 3) Final fallback mapping
                    if decoded_label is None:
                        fallback = {0: 'Vehicular Emission', 1: 'Industrial Emission', 2: 'Dust / Poor Dispersion', 3: 'Low Pollution Source'}
                        try:
                            decoded_label = fallback.get(int(src_raw), str(src_raw))
                        except Exception:
                            decoded_label = str(src_raw)

                    # Display decoded label
                    st.subheader("Predicted Main Source (Model)")
                    st.info(f"**{decoded_label}**")

                except Exception as e:
                    st.warning(f"Source model prediction failed: {e}\n{traceback.format_exc()}")

            else:
                st.info("Main source prediction skipped (either pollution not likely or source model unavailable).")

            # --- EXPLANATION: rule-based reasoning using input features ---
            st.subheader("Explanation / Dominant Factors")
            reasons = rule_based_reason(input_df.iloc[0])
            # prefer to align rule-based reason with model-decoded_label if possible
            # find if any rule-based reason label equals decoded_label, otherwise show top rule
            chosen_reason = None
            if decoded_label is not None:
                for rlabel, evidence in reasons:
                    # simple matching ignoring case and slashes
                    if rlabel.lower().split('/')[0].strip() in decoded_label.lower():
                        chosen_reason = (rlabel, evidence)
                        break
            if chosen_reason is None:
                # pick the first rule-based reason (most relevant)
                chosen_reason = reasons[0]

            # print human-readable explanation with evidence lines
            rlabel, evidence = chosen_reason
            st.write(f"**Most likely cause:** {rlabel}")
            st.markdown("**Evidence from current inputs:**")
            for ev in evidence:
                if ev:
                    st.write("- " + ev)

            # Also show a short combined statement
            combined_stmt = f"The model predicted **{decoded_label or 'N/A'}**, and based on current input values the dominant factor appears to be **{rlabel}**."
            st.markdown(f"**Summary:** {combined_stmt}")

            # show decode errors if any
            if decode_errors:
                with st.expander("Label decoding notes (expand)"):
                    st.write("\n".join(decode_errors))

    except Exception as e:
        st.error("Prediction failed: " + str(e))
        st.write(traceback.format_exc())

st.write("---")
st.caption("""
Notes:
- This app shows both the model prediction and a rule-based human explanation.
- If you want more advanced per-sample attribution (SHAP), I can prepare that next — it requires installing SHAP and may slow the app.
""")

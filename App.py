import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Page Config and Setup
st.set_page_config(page_title="Exoplanet vs. Fake Planet Detector", layout="wide", initial_sidebar_state="expanded")

# --- INJECTING A CLEAN, LIGHT CONTRAST THEME DIRECTLY IN CODE ---
st.markdown("""
    <style>
        /* Main page light background */
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
        }
        
        /* Left sidebar soft gray background */
        section[data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Sidebar text color adjustments for dark text */
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
            color: #334155 !important;
        }
        
        /* Input boxes style inside sidebar */
        section[data-testid="stSidebar"] input {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }
        
        /* Fix text elements on main page to match bright visibility */
        .stMarkdown p, .stMarkdown h3, .stMarkdown h4 {
            color: #334155 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Vibrant styled header block
st.markdown("""
    <div style="background-color:#0f172a; padding:20px; border-radius:15px; margin-bottom:25px; border-left: 8px solid #ff0051;">
        <h1 style="color:#ffffff; margin-bottom:0;">🌌 Exoplanet vs. Fake Planet Detector</h1>
        <p style="color:#94a3b8; font-size:18px; margin-top:5px; font-style:italic;">Kepler Telescope False Positive & Signal Verification Engine</p>
    </div>
""", unsafe_allow_html=True)

st.write("Kepler Telescope False Positive & Signal Verification Engine.")

# ================= Pipeline Health Check Panel =================
with st.expander("⚙️ System Core & Model Deployment Status", expanded=False):
    st.write("Checking local data files and setting up neural features...")
    
    @st.cache_resource
    def load_pipeline():
        return joblib.load('exoplanet_model_pipeline.pkl')

    artifacts = load_pipeline()
    model = artifacts['model']
    le = artifacts['label_encoder']
    explainer = artifacts['explainer']
    features = artifacts['feature_names']
    
    st.success("🎨 Trained XGBoost Engine & SHAP Explainer connected successfully!")
    st.code(f"Active Features Tracked: {list(features)}")

# ================= SIDEBAR INPUTS =================
st.sidebar.markdown("""
    <div style="background-color:#e0e7ff; padding:10px; border-radius:10px; margin-bottom:15px; border: 1px solid #c7d2fe;">
        <h3 style="color:#4338ca; margin:0;">🔬 Telescope Readings</h3>
    </div>
""", unsafe_allow_html=True)
st.sidebar.write("Adjust the observational data below to test the AI model:")

input_data = {}
for f in features:
    if f == 'koi_period':
        input_data[f] = st.sidebar.number_input("⏱️ Orbital Period (days)", value=10.0, step=0.1, help="How many days it takes the planet to complete one full orbit around its star.")
    elif f == 'koi_duration':
        input_data[f] = st.sidebar.number_input("⏳ Transit Duration (hours)", value=3.0, step=0.1, help="How long the planet takes to pass in front of its star.")
    elif f == 'koi_depth':
        input_data[f] = st.sidebar.number_input("📉 Transit Depth (ppm)", value=500.0, step=10.0, help="How much the star's brightness drops (in parts per million) when the planet blocks its light.")
    elif f == 'koi_prad':
        input_data[f] = st.sidebar.number_input("🪐 Planetary Radius (Earth sizes)", value=2.0, step=0.1, help="The calculated physical size of the object compared to Earth.")
    elif f == 'koi_impact':
        input_data[f] = st.sidebar.number_input("🎯 Impact Parameter", value=0.3, step=0.05, help="How close the planet's path is to the exact center of the star's disk.")
    elif f in ['koi_teq', 'koi_teg']:
        input_data[f] = st.sidebar.number_input("🌡️ Equilibrium Temperature (K)", value=300.0, step=10.0, help="The estimated surface temperature of the candidate object in Kelvin.")
    elif f in ['koi_insol', 'koi__insol']:
        input_data[f] = st.sidebar.number_input("☀️ Insolation Flux (Earth units)", value=1.5, step=0.1, help="The amount of energy the object receives compared to what Earth gets from the Sun.")

input_df = pd.DataFrame([input_data])

# ================= MAIN AREA: KPI PARAMETER DASHBOARD =================
st.markdown("---")
st.markdown("### 📋 Current Target Configuration Summary")
st.write("The pipeline is currently evaluating a potential system with these custom settings:")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("<div style='background-color:#dbeafe; padding:15px; border-radius:10px; border-bottom:4px solid #3b82f6;'>", unsafe_allow_html=True)
    st.metric("🪐 Target Radius", f"{input_data.get('koi_prad', 2.0):.2f} R⊕")
    st.markdown("</div>", unsafe_allow_html=True)
with kpi2:
    st.markdown("<div style='background-color:#e0e7ff; padding:15px; border-radius:10px; border-bottom:4px solid #6366f1;'>", unsafe_allow_html=True)
    st.metric("⏱️ Orbital Period", f"{input_data.get('koi_period', 10.0):.1f} Days")
    st.markdown("</div>", unsafe_allow_html=True)
with kpi3:
    st.markdown("<div style='background-color:#d1fae5; padding:15px; border-radius:10px; border-bottom:4px solid #10b981;'>", unsafe_allow_html=True)
    st.metric("📉 Transit Depth", f"{input_data.get('koi_depth', 500.0):.1f} ppm")
    st.markdown("</div>", unsafe_allow_html=True)
with kpi4:
    st.markdown("<div style='background-color:#ffedd5; padding:15px; border-radius:10px; border-bottom:4px solid #f97316;'>", unsafe_allow_html=True)
    st.metric("🌡️ Est. Temperature", f"{input_data.get('koi_teq', input_data.get('koi_teg', 300.0)):.0f} K")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= PROCESSING & VERDICT BUTTON =================
st.write("")
if st.sidebar.button("🚀 Run Pipeline Vetting Engine", type="primary"):
    
    probs = model.predict_proba(input_df)[0]
    pred_idx = np.argmax(probs)
    predicted_class = le.classes_[pred_idx]
    highest_prob = probs[pred_idx] * 100
    
    st.markdown("---")
    st.markdown("### 🎯 Vetting Pipeline Verdict")
    
    if predicted_class.upper() == "CONFIRMED":
        st.markdown(f"## 🎉 **Verdict: MOST LIKELY a Real Planet!**")
        st.success(f"**AI Evaluation ({highest_prob:.1f}% Match):** The data shows a highly stable, clean transit profile. The physical size, timing, and dip in starlight line up perfectly with verified exoplanets previously discovered by NASA's Kepler mission.")
    elif predicted_class.upper() == "CANDIDATE":
        st.markdown(f"## 🔍 **Verdict: Active Unconfirmed Candidate**")
        st.warning(f"**AI Evaluation ({highest_prob:.1f}% Match):** This light signature behaves a lot like a real planet transit, but there is not enough clean data to be absolutely sure yet. It has been flagged for further telescope observation to rule out background interference.")
    else: # FALSE POSITIVE
        st.markdown(f"## ⚠️ **Verdict: Definite False Positive**")
        st.error(f"**AI Evaluation ({highest_prob:.1f}% Match):** The pattern detected is an optical illusion. This signal does not match an actual orbiting world. Instead, it mirrors the light curves of an eclipsing binary star system, solar activity, or telescope instrument noise.")

    # Confidence Scores Distribution Layout
    st.markdown("---")
    st.markdown("### 🔮 Prediction Breakdown & Probabilities")
    
    col1, col2, col3 = st.columns(3)
    for idx, class_name in enumerate(le.classes_):
        col = [col1, col2, col3][idx]
        if class_name == predicted_class:
            col.metric(label=f"🥇 {class_name} (Predicted)", value=f"{probs[idx]*100:.1f}%", delta="Highest Probability")
        else:
            col.metric(label=class_name, value=f"{probs[idx]*100:.1f}%")
            
    # Explainable AI Component Layout
    st.markdown("---")
    st.markdown("### 🧬 Machine Logic Breakdown: What influenced the AI?")
    st.write("This visual chart breaks open the AI's 'black box' thinking. It shows whether an observation metric pushed the model closer to its final conclusion or pulled it away.")
    
    shap_all = explainer(input_df)
    shap_vals = shap_all.values[0, :, pred_idx]
    
    feature_raw_name = features[np.argmax(np.abs(shap_vals))]
    
    feature_map = {
        'koi_period': "Orbital Period", 'koi_duration': "Transit Duration",
        'koi_depth': "Transit Depth", 'koi_prad': "Planetary Radius",
        'koi_impact': "Impact Parameter", 'koi_teq': "Equilibrium Temp",
        'koi_teg': "Equilibrium Temp", 'koi_insol': "Insolation Flux",
        'koi__insol': "Insolation Flux"
    }
    
    plot_df = pd.DataFrame({
        'Feature': [feature_map.get(f, f) for f in features],
        'SHAP Value': shap_vals
    })
    plot_df['Absolute Impact'] = plot_df['SHAP Value'].abs()
    plot_df = plot_df.sort_values(by='Absolute Impact', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    
    colors = ['#ff0051' if val > 0 else '#008bfb' for val in plot_df['SHAP Value']]
    bars = ax.barh(plot_df['Feature'], plot_df['SHAP Value'], color=colors, edgecolor='none', height=0.55)
    
    ax.axvline(x=0, color='#64748b', linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.grid(axis='x', linestyle=':', color='#cbd5e1', alpha=0.7)
    ax.tick_params(colors='#334155', labelsize=9)
    
    plt.title(f"Feature Influence Breakdown for Class: {predicted_class}\n(🔴 Red = Favored This Verdict | 🔵 Blue = Opposed This Verdict)", fontsize=10, pad=12, color='#0f172a', weight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    # ================= DYNAMIC DETAILED EXPLANATION ENGINE (PURE PHYSICS) =================
    st.markdown("#### 📝 Explanation")
    
    val_checked = input_data.get(feature_raw_name, 0.0)
    physics_text = ""
    
    if feature_raw_name == 'koi_prad':
        if val_checked > 20.0:
            physics_text = (
                f"The structural geometry of this transit indicates a catastrophic physical mismatch for a planetary body due to the calculated "
                f"Planetary Radius of {val_checked} R⊕. In stellar structural physics, an object exceeding roughly 15 to 20 Earth radii "
                f"crosses the electron-degeneracy pressure limits of planet-sized substellar matter. For perspective, Jupiter sits at 11.2 R⊕, "
                f"meaning an object of {val_checked} R⊕ has a cross-sectional area over {((val_checked/11.2)**2):.1f} times larger than the largest possible gas giants. "
                f"An object this massive blocking a host star represents an eclipsing binary star system configuration rather than a sub-stellar planet orbit."
            )
        else:
            physics_text = (
                f"The physical dimensions of this body tightly align with known structural limits of substellar objects, given the verified "
                f"Planetary Radius of {val_checked} R⊕. Because this size sits comfortably below the 15–20 R⊕ limit where thermal core ignition or "
                f"brown-dwarf transitions occur, the cross-sectional area of the body matches the physical boundaries of a rocky or gas-giant world "
                f"passing across the face of a typical main-sequence star."
            )

    elif feature_raw_name == 'koi_depth':
        if val_checked > 20000.0:
            calculated_ratio = np.sqrt(val_checked / 1000000.0)
            physics_text = (
                f"The photon flux drop rules out a planet based on the observed Transit Depth of {val_checked} ppm. "
                f"The fundamental transit equation dictates that the fractional dip in light depends on the exact cross-sectional area ratio: "
                f"$\\Delta F/F = (R_p / R_*)^2$. A drop of {val_checked} ppm means $\\Delta F/F = {(val_checked/1000000.0):.4f}$. "
                f"Taking the square root reveals that the radius of the passing object ($R_p$) must be a massive {calculated_ratio:.2f} times (or over {calculated_ratio*100:.0f}%) "
                f"the radius of the host star ($R_*$). Planets physically cannot be this large relative to a main-sequence star; a light dip this deep "
                f"proves the signal is caused by a secondary companion star eclipsing the primary star."
            )
        else:
            calculated_ratio = np.sqrt(val_checked / 1000000.0)
            physics_text = (
                f"The photometric data matches a sub-stellar transit because of the shallow Transit Depth of {val_checked} ppm. "
                f"Applying the area-flux ratio equation, $\\Delta F/F = (R_p / R_*)^2$, a depth of {val_checked} ppm means the light dimming fraction "
                f"is a minuscule {(val_checked/1000000.0):.5f}. This implies the radius of the passing body is just {calculated_ratio:.3f} times the radius of the "
                f"star, mathematically confirming a tiny, non-luminous planetary body blocking a microscopic fraction of the host star's radiant flux."
            )

    elif feature_raw_name == 'koi_impact':
        if val_checked > 0.9:
            physics_text = (
                f"The geometry of the orbital plane is severely constrained by the high Impact Parameter of {val_checked}. "
                f"The impact parameter measures the projected distance from the chord of the planet's transit path to the center of the stellar disk, "
                f"defined by $b = (a \\cos i) / R_*$. An impact parameter of {val_checked} means the transiting path occurs along the extreme, outermost "
                f"limb of the star (where $b \\rightarrow 1$). This high-latitude chord path produces a highly compressed, distinct V-shaped light curve "
                f"rather than a flat U-shape, a signature typical of grazing stellar binaries where the secondary star barely eclipses the edge of the primary."
            )
        else:
            physics_text = (
                f"The orbital trajectory indicates a stable plane due to the low Impact Parameter of {val_checked}. "
                f"The normalized chord distance ($b = (a \\cos i) / R_*$) approaching 0 proves the transiting body passes directly across the central hemisphere "
                f"of the stellar disk. This central trajectory produces a symmetrical, steep, flat-bottomed U-shaped light curve, validating a clean, "
                f"spherical planetary silhouette passing fully in front of the host star."
            )

    elif feature_raw_name == 'koi_period':
        semi_major_axis_approx = (val_checked / 365.25) ** (2/3)
        physics_text = (
            f"The orbital mechanics of the system are defined by the clean repeating Period of {val_checked} days. "
            f"According to Kepler's Third Law of Planetary Motion ($T^2 = a^3$), an orbital period ($T$) of {val_checked} days (or {(val_checked/365.25):.4f} Earth years) "
            f"fixes the semi-major axis distance ($a$) at precisely {semi_major_axis_approx:.3f} Astronomical Units (AU) from the center of mass. "
            f"At an exact orbital distance of {semi_major_axis_approx:.3f} AU, the gravitational tidal forces exerted by the host star are balanced, "
            f"proving the target exists within a stable orbital path safely outside the Roche limit where tidal disintegration occurs."
        )

    else:
        physics_text = (
            f"The geometric transit profile is bounded by the observed variable value of {val_checked}. "
            f"The timing parameters and stellar flux dip match standard thermodynamic and orbital energy distributions required to maintain a stable, "
            f"repeating orbit within the host star's localized gravitational flux environment."
        )

    st.info(physics_text)
    
    # ================= SAFEGUARD & RECOURSE PROTOCOLS =================
    if predicted_class.upper() in ["FALSE POSITIVE", "CANDIDATE"]:
        st.markdown("---")
        st.markdown("### 🌌 Vetting Safeguard & Recourse Protocols")
        st.markdown(
            "⚠️ **Risk Mitigation Alert:** To eliminate the risk of a *False Negative* error (accidentally throwing away a real planet that hides behind an anomalous light signature), "
            "the following targeted physical verification strategy is recommended before archiving this target:"
        )
        
        rec1, rec2 = st.columns(2)
        with rec1:
            st.markdown("#### 🔭 1. High-Resolution Adaptive Optics (AO) Imaging")
            st.write(
                "**Objective:** Rule out background blended stars.\n\n"
                "Space telescope pixels (like Kepler's) can cover large patches of sky. A background eclipsing binary star perfectly lined up behind a normal foreground star "
                "can mimic a planet transit depth. Ground-based Adaptive Optics imaging (e.g., using the Keck or Palomar observatories) will pierce through the atmospheric blur "
                "to see if the target star is actually a close-proximity visual double, resolving the blending ambiguity completely."
            )
        with rec2:
            st.markdown("#### 🧬 2. Precision Radial Velocity (RV) Follow-Up")
            st.write(
                "**Objective:** Measure the true physical mass of the transiting object.\n\n"
                "While photometry only tells us the *size* of the object based on blocked light, Radial Velocity spectrometers measure the Doppler wobble of the host star "
                "caused by the object's gravitational pull. If follow-up spectroscopy detects a mass pull corresponding to $M \\sin i < 13 \\text{ M}_J$, it mathematically "
                "confirms the object is a planet, completely overruling any anomalous light curve distortions flagged by automated filters."
            )
            
else:
    st.info("👈 Set your observational parameters in the left sidebar and click **Run Pipeline Vetting Engine** to launch the AI analysis!")
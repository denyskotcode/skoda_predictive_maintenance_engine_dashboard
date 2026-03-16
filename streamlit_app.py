"""
Škoda Predictive Maintenance System — Streamlit Dashboard
AI-powered engine failure prediction and diagnostics.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Škoda Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Škoda brand colors
# ---------------------------------------------------------------------------
SKODA_GREEN = "#4EA94B"
SKODA_DARK_GREEN = "#2C5F2D"
SKODA_LIGHT_GREEN = "#A8D5A2"
COLOR_SCALE = [[0, SKODA_LIGHT_GREEN], [0.5, SKODA_GREEN], [1.0, SKODA_DARK_GREEN]]

# ---------------------------------------------------------------------------
# Global CSS styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
        border: 1px solid #4EA94B;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 13px;
        color: #555;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #2C5F2D;
    }
    .risk-badge-low {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #4EA94B;
        padding: 8px 18px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
        padding: 8px 18px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
        padding: 8px 18px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .sidebar-section {
        background: #f0f8f0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    h1 { color: #2C5F2D; }
    h2 { color: #2C5F2D; }
    h3 { color: #4EA94B; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ai4i2020.csv')


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all trained model artifacts from disk."""
    def load_pkl(name):
        with open(os.path.join(MODELS_DIR, name), 'rb') as f:
            return pickle.load(f)

    binary_clf = load_pkl('binary_classifier.pkl')
    multi_clf = load_pkl('multiclass_classifier.pkl')
    label_enc = load_pkl('label_encoder.pkl')
    scaler = load_pkl('scaler.pkl')
    feature_names = load_pkl('feature_names.pkl')
    model_comparison = load_pkl('model_comparison.pkl')
    shap_explainer = load_pkl('shap_explainer.pkl')

    return binary_clf, multi_clf, label_enc, scaler, feature_names, model_comparison, shap_explainer


@st.cache_data
def load_dataset():
    """Load raw dataset for exploration tab."""
    df = pd.read_csv(DATA_PATH)
    # Rename to snake_case
    rename_map = {
        'Air temperature [K]': 'air_temp_k',
        'Process temperature [K]': 'process_temp_k',
        'Rotational speed [rpm]': 'rotational_speed_rpm',
        'Torque [Nm]': 'torque_nm',
        'Tool wear [min]': 'tool_wear_min',
        'Machine failure': 'machine_failure',
        'Type': 'type'
    }
    df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Feature engineering (must match train.py exactly)
# ---------------------------------------------------------------------------
def engineer_input_features(type_val: str, air_temp: float, process_temp: float,
                             rpm: int, torque: float, tool_wear: int) -> pd.DataFrame:
    """
    Apply the same feature engineering pipeline used during training.
    Returns a single-row DataFrame ready for scaling and prediction.
    """
    temp_diff_k = process_temp - air_temp
    power_w = torque * rpm * 2 * np.pi / 60
    torque_wear_product = torque * tool_wear
    speed_torque_ratio = rpm / (torque + 1e-6)
    is_low_quality = 1 if type_val == 'L' else 0
    is_high_quality = 1 if type_val == 'H' else 0

    return pd.DataFrame([{
        'air_temp_k': air_temp,
        'process_temp_k': process_temp,
        'rotational_speed_rpm': rpm,
        'torque_nm': torque,
        'tool_wear_min': tool_wear,
        'temp_diff_k': temp_diff_k,
        'power_w': power_w,
        'torque_wear_product': torque_wear_product,
        'speed_torque_ratio': speed_torque_ratio,
        'is_low_quality': is_low_quality,
        'is_high_quality': is_high_quality
    }])


# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
try:
    (binary_clf, multi_clf, label_enc, scaler,
     feature_names, model_comparison, shap_explainer) = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🔧 Škoda Predictive Maintenance System")
st.markdown("**AI-powered engine failure prediction & diagnostics**")
st.markdown(
    "_Real-time sensor monitoring system that predicts equipment failures before they occur. "
    "Built on the AI4I 2020 dataset with 10,000 manufacturing records and 5 failure mode categories._"
)
st.markdown("---")

if not models_loaded:
    st.error(f"❌ Models not found. Please run `python train.py` first.\n\nError: {load_error}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — Sensor Readings Input
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Sensor Readings")
st.sidebar.markdown("_Simulate real-time sensor input from the production line_")

# Session state defaults
if 'air_temp' not in st.session_state:
    st.session_state.air_temp = 300.0
if 'process_temp' not in st.session_state:
    st.session_state.process_temp = 310.0
if 'rpm' not in st.session_state:
    st.session_state.rpm = 1500
if 'torque' not in st.session_state:
    st.session_state.torque = 40.0
if 'tool_wear' not in st.session_state:
    st.session_state.tool_wear = 100
if 'product_type' not in st.session_state:
    st.session_state.product_type = 'Low (L)'

# Simulation buttons
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🎲 Random State", use_container_width=True):
        st.session_state.air_temp = round(float(np.random.uniform(295, 305)), 1)
        st.session_state.process_temp = round(float(np.random.uniform(305, 315)), 1)
        st.session_state.rpm = int(np.random.randint(1000, 2901))
        st.session_state.torque = round(float(np.random.uniform(3, 80)), 1)
        st.session_state.tool_wear = int(np.random.randint(0, 251))
        st.session_state.product_type = np.random.choice(['Low (L)', 'Medium (M)', 'High (H)'])

with col_btn2:
    if st.button("⚠️ Critical State", use_container_width=True):
        # High torque, high tool wear, low rpm → likely failure
        st.session_state.air_temp = 304.5
        st.session_state.process_temp = 312.5
        st.session_state.rpm = 1100
        st.session_state.torque = 72.0
        st.session_state.tool_wear = 235
        st.session_state.product_type = 'Low (L)'

st.sidebar.markdown("---")

# Product quality selectbox
type_label_map = {'Low (L)': 'L', 'Medium (M)': 'M', 'High (H)': 'H'}
product_type_label = st.sidebar.selectbox(
    "Product Quality",
    options=['Low (L)', 'Medium (M)', 'High (H)'],
    index=['Low (L)', 'Medium (M)', 'High (H)'].index(st.session_state.product_type),
    key='product_type'
)
product_type = type_label_map[product_type_label]

# Sensor sliders
air_temp = st.sidebar.slider(
    "Air Temperature (K)", min_value=295.0, max_value=305.0,
    value=st.session_state.air_temp, step=0.1, key='air_temp'
)
process_temp = st.sidebar.slider(
    "Process Temperature (K)", min_value=305.0, max_value=315.0,
    value=st.session_state.process_temp, step=0.1, key='process_temp'
)
rpm = st.sidebar.slider(
    "Rotational Speed (rpm)", min_value=1000, max_value=2900,
    value=st.session_state.rpm, step=10, key='rpm'
)
torque = st.sidebar.slider(
    "Torque (Nm)", min_value=3.0, max_value=80.0,
    value=st.session_state.torque, step=0.1, key='torque'
)
tool_wear = st.sidebar.slider(
    "Tool Wear (min)", min_value=0, max_value=250,
    value=st.session_state.tool_wear, step=1, key='tool_wear'
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Calculated Power:** `{torque * rpm * 2 * np.pi / 60:.0f} W`  \n"
    f"**Temp Differential:** `{process_temp - air_temp:.1f} K`"
)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
X_input = engineer_input_features(product_type, air_temp, process_temp, rpm, torque, tool_wear)

# Reorder columns to match training feature order
X_input = X_input[feature_names]
X_scaled = scaler.transform(X_input)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Binary prediction
failure_prob = binary_clf.predict_proba(X_scaled)[0][1]
failure_pred = int(failure_prob >= 0.5)

# Multiclass prediction
multi_pred_enc = multi_clf.predict(X_scaled)[0]
multi_proba = multi_clf.predict_proba(X_scaled)[0]
multi_pred_label = label_enc.inverse_transform([multi_pred_enc])[0]
multi_confidence = multi_proba.max()
class_names = label_enc.classes_.tolist()

# Calculated power
power_w = torque * rpm * 2 * np.pi / 60
power_delta = power_w - 2860


# ---------------------------------------------------------------------------
# Top KPI metrics
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🚦 Failure Risk")
    pct = failure_prob * 100
    if failure_prob < 0.3:
        badge = f'<span class="risk-badge-low">✅ LOW RISK</span>'
    elif failure_prob < 0.7:
        badge = f'<span class="risk-badge-medium">⚠️ MEDIUM RISK</span>'
    else:
        badge = f'<span class="risk-badge-high">🚨 HIGH RISK</span>'
    st.markdown(f"<div style='text-align:center; font-size:36px; font-weight:700; color:#2C5F2D'>{pct:.1f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center; margin-top:6px'>{badge}</div>", unsafe_allow_html=True)

with col2:
    st.markdown("#### 🔍 Failure Mode")
    mode_colors = {
        'No Failure': '✅', 'TWF': '🔴', 'HDF': '🟠',
        'PWF': '🟡', 'OSF': '🔴', 'RNF': '🟣'
    }
    icon = mode_colors.get(multi_pred_label, '❓')
    st.markdown(
        f"<div style='text-align:center; font-size:28px; font-weight:700; color:#2C5F2D'>"
        f"{icon} {multi_pred_label}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='text-align:center; color:#555; margin-top:6px'>Confidence: {multi_confidence*100:.1f}%</div>",
        unsafe_allow_html=True
    )

with col3:
    st.markdown("#### ⚡ Power Output")
    delta_str = f"+{power_delta:.0f} W" if power_delta >= 0 else f"{power_delta:.0f} W"
    delta_color = "normal" if abs(power_delta) < 1000 else "inverse"
    st.metric(
        label="",
        value=f"{power_w:.0f} W",
        delta=delta_str,
        delta_color=delta_color
    )
    st.caption("vs. nominal 2,860 W")

# Progress bar for risk
st.markdown("**Failure Probability**")
progress_val = int(failure_prob * 100)
if failure_prob < 0.3:
    bar_color = SKODA_GREEN
elif failure_prob < 0.7:
    bar_color = "#ffc107"
else:
    bar_color = "#dc3545"

st.markdown(
    f"""<div style="background-color:#e9ecef; border-radius:10px; height:20px; margin-bottom:20px;">
    <div style="background-color:{bar_color}; width:{progress_val}%; height:100%; border-radius:10px;
    transition: width 0.5s ease; display:flex; align-items:center; justify-content:center;">
    <span style="color:white; font-size:12px; font-weight:700;">{progress_val}%</span>
    </div></div>""",
    unsafe_allow_html=True
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 SHAP Explanation",
    "📈 Model Performance",
    "🔧 Failure Mode Analysis",
    "📊 Sensor Data Explorer",
    "📋 About"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: SHAP Explanation
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Why this prediction? Feature contributions")
    st.caption("Features pushing toward failure (red) vs. safety (green)")

    try:
        # Import shap inside function to avoid heavy top-level import
        from src.explainability import get_shap_for_single_prediction

        shap_vals = get_shap_for_single_prediction(shap_explainer, X_scaled_df)

        # Build DataFrame for plotting
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_vals,
            'abs_shap': np.abs(shap_vals)
        }).sort_values('abs_shap', ascending=True)

        # Color bars: red = increases risk, green = decreases risk
        colors = [SKODA_DARK_GREEN if v < 0 else "#dc3545" for v in shap_df['shap_value']]

        fig_shap = go.Figure(go.Bar(
            x=shap_df['shap_value'],
            y=shap_df['feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.4f}" for v in shap_df['shap_value']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>'
        ))
        fig_shap.update_layout(
            xaxis_title="SHAP Value (impact on failure probability)",
            yaxis_title="Feature",
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=13),
            xaxis=dict(
                gridcolor='#e8f5e8',
                zerolinecolor=SKODA_GREEN,
                zerolinewidth=2
            ),
            margin=dict(l=160, r=80, t=20, b=60)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Text explanation — top 3 risk factors
        risk_factors = shap_df[shap_df['shap_value'] > 0].sort_values('abs_shap', ascending=False).head(3)
        safety_factors = shap_df[shap_df['shap_value'] < 0].sort_values('abs_shap', ascending=False).head(3)

        if len(risk_factors) > 0:
            risk_list = ', '.join([
                f"**{r['feature']}** ({r['shap_value']:+.4f})"
                for _, r in risk_factors.iterrows()
            ])
            st.info(f"🔴 **Top risk factors:** {risk_list}")
        if len(safety_factors) > 0:
            safe_list = ', '.join([
                f"**{r['feature']}** ({r['shap_value']:+.4f})"
                for _, r in safety_factors.iterrows()
            ])
            st.success(f"✅ **Top safety factors:** {safe_list}")

    except Exception as e:
        st.error(f"SHAP explanation unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Model Performance
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Comparison — Binary Failure Classification")

    metrics = model_comparison['metrics']
    roc_curves = model_comparison['roc_curves']
    pr_curves = model_comparison['pr_curves']
    best_name = model_comparison['best_model_name']

    col_roc, col_pr = st.columns(2)

    # ROC Curves
    model_colors = ['#4EA94B', '#2C5F2D', '#e07b39', '#6e5fa6']
    with col_roc:
        fig_roc = go.Figure()
        for (name, (fpr, tpr)), color in zip(roc_curves.items(), model_colors):
            auc = metrics[name]['roc_auc']
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{name} (AUC={auc:.3f})",
                line=dict(color=color, width=2.5 if name == best_name else 1.5)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='grey'),
            name='Random', showlegend=False
        ))
        fig_roc.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380,
            plot_bgcolor='white',
            legend=dict(x=0.4, y=0.1),
            xaxis=dict(gridcolor='#e8f5e8'),
            yaxis=dict(gridcolor='#e8f5e8')
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall Curves
    with col_pr:
        fig_pr = go.Figure()
        for (name, (prec, rec)), color in zip(pr_curves.items(), model_colors):
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode='lines',
                name=name,
                line=dict(color=color, width=2.5 if name == best_name else 1.5)
            ))
        fig_pr.update_layout(
            title="Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=380,
            plot_bgcolor='white',
            legend=dict(x=0.5, y=0.8),
            xaxis=dict(gridcolor='#e8f5e8'),
            yaxis=dict(gridcolor='#e8f5e8')
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Metrics table
    st.markdown("#### Metrics Summary")
    table_data = []
    for name, m in metrics.items():
        table_data.append({
            'Model': f"⭐ {name}" if name == best_name else name,
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision (Failure)': f"{m['precision_failure']:.4f}",
            'Recall (Failure)': f"{m['recall_failure']:.4f}",
            'F1 (Failure)': f"{m['f1_failure']:.4f}",
            'ROC AUC': f"{m['roc_auc']:.4f}"
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Confusion Matrix for best model
    st.markdown(f"#### Confusion Matrix — {best_name}")
    cm = model_comparison['confusion_matrix']
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=['Predicted: OK', 'Predicted: Failure'],
        y=['Actual: OK', 'Actual: Failure'],
        colorscale=[[0, '#f0f8f0'], [1, SKODA_DARK_GREEN]],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(size=20),
        showscale=True
    ))
    fig_cm.update_layout(
        height=350,
        width=450,
        xaxis=dict(side='bottom'),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    col_cm, _ = st.columns([1, 1])
    with col_cm:
        st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Failure Mode Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Failure Mode Analysis")

    col_bar, col_pie = st.columns(2)

    # Probabilities for current input
    with col_bar:
        st.markdown("#### Failure Probabilities (Current Input)")
        multi_proba_arr = multi_clf.predict_proba(X_scaled)[0]
        proba_df = pd.DataFrame({
            'Failure Mode': class_names,
            'Probability': multi_proba_arr
        }).sort_values('Probability', ascending=True)

        bar_colors = [
            f"rgba({int(p*200)}, {int((1-p)*180)}, 50, 0.85)"
            for p in proba_df['Probability']
        ]
        fig_proba = go.Figure(go.Bar(
            x=proba_df['Probability'],
            y=proba_df['Failure Mode'],
            orientation='h',
            marker_color=bar_colors,
            text=[f"{p*100:.1f}%" for p in proba_df['Probability']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Probability: %{x:.4f}<extra></extra>'
        ))
        fig_proba.update_layout(
            xaxis_title="Probability",
            xaxis=dict(range=[0, min(1.2, proba_df['Probability'].max() * 1.4)], gridcolor='#e8f5e8'),
            height=350,
            plot_bgcolor='white',
            margin=dict(l=20, r=80, t=20, b=40)
        )
        st.plotly_chart(fig_proba, use_container_width=True)

    # Dataset distribution pie
    with col_pie:
        st.markdown("#### Failure Mode Distribution (Dataset)")
        try:
            df_raw = load_dataset()
            failure_mask = df_raw['machine_failure'] == 1
            df_fail = df_raw[failure_mask].copy()

            def get_type(row):
                for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
                    if col in row and row[col] == 1:
                        return col
                return 'Unknown'

            if failure_mask.sum() > 0:
                df_fail['mode'] = df_fail.apply(get_type, axis=1)
                counts = df_fail['mode'].value_counts().reset_index()
                counts.columns = ['mode', 'count']
                fig_pie = px.pie(
                    counts, values='count', names='mode',
                    color_discrete_sequence=[SKODA_GREEN, SKODA_DARK_GREEN, '#e07b39', '#6e5fa6', '#e04a5b']
                )
                fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)
        except Exception:
            st.info("Dataset not available for pie chart.")

    # Failure mode descriptions
    st.markdown("#### Failure Mode Reference")
    failure_info = {
        "🔧 TWF — Tool Wear Failure": (
            "Occurs when tool wear exceeds 200–240 min threshold. "
            "The cutting tool degrades beyond acceptable limits, causing dimensional inaccuracies and surface defects."
        ),
        "🌡️ HDF — Heat Dissipation Failure": (
            "Occurs when temp difference between process and air < 8.6 K AND rotational speed < 1380 rpm. "
            "Insufficient cooling at low speeds causes thermal stress and component warping."
        ),
        "⚡ PWF — Power Failure": (
            "Occurs when mechanical power < 3,500 W or > 9,000 W. "
            "Operating outside the optimal power envelope causes motor strain or energy waste."
        ),
        "💪 OSF — Overstrain Failure": (
            "Occurs when torque × tool wear exceeds a quality-dependent threshold (L: 11K, M: 12K, H: 13K). "
            "Mechanical overstrain damages bearings, couplings, and structural components."
        ),
        "🎲 RNF — Random Failure": (
            "0.1% random failure probability regardless of operating parameters. "
            "Models stochastic failure events like unexpected material defects or random component failures."
        )
    }
    for title, desc in failure_info.items():
        with st.expander(title):
            st.write(desc)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Sensor Data Explorer
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Sensor Data Explorer")

    try:
        df_raw = load_dataset()

        sensor_cols = [
            ('air_temp_k', 'Air Temp (K)', air_temp),
            ('process_temp_k', 'Process Temp (K)', process_temp),
            ('rotational_speed_rpm', 'Rotational Speed (rpm)', rpm),
            ('torque_nm', 'Torque (Nm)', torque),
            ('tool_wear_min', 'Tool Wear (min)', tool_wear),
        ]

        # Add power_w to the dataset for exploration
        df_raw['power_w'] = (df_raw['torque_nm'] * df_raw['rotational_speed_rpm'] *
                              2 * np.pi / 60)
        sensor_cols.append(('power_w', 'Power (W)', power_w))

        # 6 histograms in 2×3 grid
        st.markdown("#### Sensor Distributions (with current values)")
        fig_hist = make_subplots(rows=2, cols=3, subplot_titles=[c[1] for c in sensor_cols])

        for idx, (col_name, col_label, user_val) in enumerate(sensor_cols):
            row = idx // 3 + 1
            col = idx % 3 + 1
            if col_name not in df_raw.columns:
                continue
            data_ok = df_raw[df_raw['machine_failure'] == 0][col_name].dropna()
            data_fail = df_raw[df_raw['machine_failure'] == 1][col_name].dropna()

            fig_hist.add_trace(go.Histogram(
                x=data_ok, name='OK', marker_color=SKODA_LIGHT_GREEN,
                opacity=0.7, nbinsx=30, showlegend=(idx == 0)
            ), row=row, col=col)
            fig_hist.add_trace(go.Histogram(
                x=data_fail, name='Failure', marker_color='#dc3545',
                opacity=0.7, nbinsx=30, showlegend=(idx == 0)
            ), row=row, col=col)
            fig_hist.add_vline(
                x=user_val, line_dash="dash", line_color=SKODA_DARK_GREEN,
                line_width=2, row=row, col=col
            )

        fig_hist.update_layout(
            height=500, barmode='overlay',
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=True,
            legend=dict(x=0.85, y=0.95)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Scatter plot explorer
        st.markdown("#### Scatter Plot Explorer")
        numeric_cols = [c[0] for c in sensor_cols if c[0] in df_raw.columns]
        col_x, col_y = st.columns(2)
        with col_x:
            x_feat = st.selectbox("X axis", options=numeric_cols, index=2)
        with col_y:
            y_feat = st.selectbox("Y axis", options=numeric_cols, index=3)

        user_x = dict(zip([c[0] for c in sensor_cols], [c[2] for c in sensor_cols])).get(x_feat, 0)
        user_y = dict(zip([c[0] for c in sensor_cols], [c[2] for c in sensor_cols])).get(y_feat, 0)

        scatter_sample = df_raw.sample(min(2000, len(df_raw)), random_state=42)
        fig_scatter = px.scatter(
            scatter_sample, x=x_feat, y=y_feat,
            color=scatter_sample['machine_failure'].map({0: 'OK', 1: 'Failure'}),
            color_discrete_map={'OK': SKODA_LIGHT_GREEN, 'Failure': '#dc3545'},
            opacity=0.5,
            labels={x_feat: x_feat, y_feat: y_feat}
        )
        # Add current user point
        fig_scatter.add_trace(go.Scatter(
            x=[user_x], y=[user_y],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star', line=dict(color='white', width=2)),
            name='Current Input'
        ))
        fig_scatter.update_layout(
            height=400, plot_bgcolor='white',
            xaxis=dict(gridcolor='#e8f5e8'),
            yaxis=dict(gridcolor='#e8f5e8')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Correlation matrix
        st.markdown("#### Correlation Matrix")
        corr_cols = [c[0] for c in sensor_cols if c[0] in df_raw.columns] + ['machine_failure']
        corr_matrix = df_raw[corr_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale=[[0, '#dc3545'], [0.5, 'white'], [1.0, SKODA_DARK_GREEN]],
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
            texttemplate="%{text}",
            textfont=dict(size=11)
        ))
        fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"Dataset explorer error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: About
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### About This Project")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
**Škoda Predictive Maintenance System** is a proof-of-concept AI application demonstrating
real-time failure prediction for automotive manufacturing equipment.

The system monitors sensor readings from a CNC machine (air/process temperature, rotational speed,
torque, tool wear) and predicts whether a failure will occur and what type of failure is most likely.

**Key capabilities:**
- Binary failure risk assessment with probability score
- Multi-class failure mode identification (5 types)
- Explainable AI via SHAP values for each prediction
- Interactive sensor data exploration
- Model comparison across 4 ML algorithms
        """)

    with col_b:
        st.markdown("#### Technology Stack")
        tech_items = [
            ("🐍 Python 3.10+", "Core language"),
            ("🌐 Streamlit", "Web dashboard"),
            ("🤖 scikit-learn", "ML pipeline"),
            ("⚡ XGBoost", "Gradient boosting"),
            ("💡 LightGBM", "Fast gradient boosting"),
            ("🔍 SHAP", "Model explainability"),
            ("📊 Plotly", "Interactive charts"),
            ("🐼 Pandas / NumPy", "Data processing"),
        ]
        for name, desc in tech_items:
            st.markdown(f"- **{name}** — {desc}")

    st.markdown("---")
    st.markdown("#### Dataset")
    st.markdown("""
**AI4I 2020 Predictive Maintenance Dataset**
- Source: UCI Machine Learning Repository (ID: 601)
- Size: 10,000 records, 14 features
- Failure rate: ~3.4% (339 failures)
- License: CC BY 4.0
- Citation: Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications.
  _2020 Third International Conference on Artificial Intelligence for Industries (AI4I)_
    """)

    st.markdown("---")
    st.markdown("#### Best Model Performance")
    best_metrics = model_comparison['metrics'][best_name]
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
    col_m2.metric("Precision (Failure)", f"{best_metrics['precision_failure']:.4f}")
    col_m3.metric("Recall (Failure)", f"{best_metrics['recall_failure']:.4f}")
    col_m4.metric("ROC AUC", f"{best_metrics['roc_auc']:.4f}")
    st.caption(f"Best model: **{best_name}**")

    st.markdown("---")
    st.markdown("""
#### Links
- 🐙 [GitHub Repository](https://github.com/YOUR-USERNAME/skoda-predictive-maintenance) _(update with your URL)_
- 📓 [Kaggle Notebook](https://www.kaggle.com/your-username/skoda-predictive-maintenance) _(update with your URL)_
- 🌐 [Live Demo](https://your-app.streamlit.app) _(update after deployment)_
    """)

    st.markdown("---")
    st.caption("Built as a portfolio project for ML Engineer position applications. "
               "Data: AI4I 2020 (CC BY 4.0). Code: MIT License.")

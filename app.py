"""
app.py — INDUSTRIA Unified Industrial AI Platform
NOVA-Ω FEM + SensorGuard + InfraGuard + FEM Bridge
Stack : LSTM AE · SymPy · Plotly · Streamlit Cloud
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="INDUSTRIA — Industrial AI Platform",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html,body,[class*="css"]{ font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:#0A0E17; }

[data-testid="stSidebar"] { background:#060A12 !important; border-right:1px solid #1A2535; }
[data-testid="stSidebar"] * { color:#8BA0BC !important; }
[data-testid="stSidebar"] h3 { color:#00D4FF !important; font-size:10px !important;
    letter-spacing:3px; font-family:'IBM Plex Mono',monospace !important; }

[data-testid="metric-container"] { background:#0D1420; border:1px solid #1A2535; border-radius:0; }
[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace !important;
    color:#00D4FF !important; font-size:24px !important; }
[data-testid="stMetricLabel"] { color:#3A5070 !important; font-size:10px !important;
    letter-spacing:3px; text-transform:uppercase; }

h1 { font-family:'IBM Plex Mono',monospace !important; color:#F0F6FF !important;
     font-size:28px !important; font-weight:500 !important; letter-spacing:2px; }
h2 { font-family:'IBM Plex Mono',monospace !important; color:#00D4FF !important;
     font-size:14px !important; font-weight:400 !important; letter-spacing:3px;
     text-transform:uppercase; }
h3 { font-family:'IBM Plex Mono',monospace !important; color:#4A7090 !important;
     font-size:10px !important; letter-spacing:3px; text-transform:uppercase; }

[data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace !important;
    font-size:10px !important; letter-spacing:2px; color:#4A7090 !important; }
[aria-selected="true"] { color:#00D4FF !important; border-bottom:2px solid #00D4FF !important; }

.stButton>button { background:transparent; border:1px solid #1A3550; color:#00D4FF;
    font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:2px;
    border-radius:0; padding:10px 20px; }
.stButton>button:hover { border-color:#00D4FF; background:rgba(0,212,255,0.05); }

.status-nominal  { color:#39FF14; font-family:'IBM Plex Mono',monospace; font-size:12px; }
.status-medium   { color:#FFB800; font-family:'IBM Plex Mono',monospace; font-size:12px; }
.status-high     { color:#FF6B35; font-family:'IBM Plex Mono',monospace; font-size:12px; }
.status-critical { color:#FF2020; font-family:'IBM Plex Mono',monospace; font-size:12px;
                   animation:blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.kpi-row { display:flex; gap:12px; margin:8px 0; }
.kpi-box { background:#0D1420; border:1px solid #1A2535; padding:12px 16px; flex:1; }
.kpi-val { font-family:'IBM Plex Mono',monospace; font-size:22px; color:#00D4FF; }
.kpi-lbl { font-size:10px; color:#3A5070; letter-spacing:2px; text-transform:uppercase; margin-top:4px; }

.diag-box-NOMINAL  { background:rgba(57,255,20,0.05);  border-left:3px solid #39FF14;
    padding:10px 14px; margin:4px 0; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#39FF14; }
.diag-box-LOW      { background:rgba(0,212,255,0.05);  border-left:3px solid #00D4FF;
    padding:10px 14px; margin:4px 0; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#00D4FF; }
.diag-box-MEDIUM   { background:rgba(255,184,0,0.05);  border-left:3px solid #FFB800;
    padding:10px 14px; margin:4px 0; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#FFB800; }
.diag-box-HIGH     { background:rgba(255,107,53,0.05); border-left:3px solid #FF6B35;
    padding:10px 14px; margin:4px 0; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#FF6B35; }
.diag-box-CRITICAL { background:rgba(255,32,32,0.08);  border-left:3px solid #FF2020;
    padding:10px 14px; margin:4px 0; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#FF2020; }

.section-header { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#1A3550;
    letter-spacing:4px; text-transform:uppercase; border-bottom:1px solid #1A2535;
    padding-bottom:6px; margin:20px 0 12px; }
hr { border-color:#1A2535 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    plot_bgcolor="#0A0E17", paper_bgcolor="#0A0E17",
    font=dict(family="IBM Plex Mono", color="#8BA0BC", size=10),
    xaxis=dict(gridcolor="#111827", linecolor="#1A2535"),
    yaxis=dict(gridcolor="#111827", linecolor="#1A2535"),
    margin=dict(l=0, r=0, t=30, b=0),
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='text-align:center;padding:16px 0;'>
  <div style='font-family:"IBM Plex Mono",monospace;font-size:28px;color:#00D4FF;'>⚙</div>
  <div style='font-family:"IBM Plex Mono",monospace;font-size:13px;color:#F0F6FF;
              letter-spacing:4px;margin-top:4px;'>INDUSTRIA</div>
  <div style='font-family:"IBM Plex Mono",monospace;font-size:9px;color:#1A3550;
              letter-spacing:3px;margin-top:2px;'>INDUSTRIAL AI PLATFORM</div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### MODULE")
    module = st.selectbox("Module", [
        "⬡  SensorGuard — Process Sensors",
        "⬡  InfraGuard — Water Infrastructure",
        "⬡  FEM Bridge — Structural Health",
    ])

    st.markdown("---")
    st.markdown("### PARAMETERS")

    if "SensorGuard" in module:
        dataset  = st.selectbox("Dataset",  ["SMAP", "MSL"])
        channel  = st.selectbox("Channel",  ["P-1","P-2","P-3","E-1","T-1"] if dataset=="SMAP"
                                             else ["M-1","M-2","M-3","M-6","C-1"])
        thr_pct  = st.slider("Threshold percentile", 80, 99, 94)
        window   = st.slider("LSTM window", 16, 128, 64, 8)
        epochs   = st.slider("Epochs", 10, 50, 30, 5)
        viz_rng  = st.slider("Display range (pts)", 200, 5000, 1500, 100)

    elif "InfraGuard" in module:
        process  = st.selectbox("Process", ["All 6 processes","P1: Raw intake",
                                             "P2: Pre-treatment","P3: Ultrafiltration",
                                             "P4: De-chlorination","P5: RO","P6: Product"])
        thr_pct  = st.slider("Threshold percentile", 80, 99, 95)
        window   = st.slider("LSTM window", 32, 128, 96, 16)
        epochs   = st.slider("Epochs", 10, 40, 25, 5)
        viz_rng  = st.slider("Display range (pts)", 500, 6000, 2000, 100)

    else:  # FEM Bridge
        damage   = st.slider("Simulated damage level", 0.0, 0.40, 0.0, 0.05,
                              help="0=healthy, 0.40=severe structural damage")
        t_sec    = st.select_slider("Signal duration (s)", [600,1800,3600,7200], 3600)
        n_modes  = st.slider("Modes to extract", 4, 6, 6)

    st.markdown("---")
    run_btn = st.button("▶  RUN ANALYSIS", width="stretch")

    st.markdown("""
<div style='font-family:"IBM Plex Mono",monospace;font-size:9px;color:#1A3550;
            line-height:2.2;padding-top:8px;'>
ENGINE · LSTM Autoencoder<br>
FEM  &nbsp;&nbsp;· NOVA-Ω pipeline<br>
DATA &nbsp;&nbsp;· NASA SMAP/MSL · SWaT · Norway<br>
VER  &nbsp;&nbsp;· INDUSTRIA v1.0
</div>""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
col_t, col_s = st.columns([3, 1])
with col_t:
    mod_label = module.split("—")[1].strip() if "—" in module else module
    st.markdown(f"""
<h1>INDUSTRIA</h1>
<p style='font-family:"IBM Plex Mono",monospace;font-size:11px;color:#3A5070;
           letter-spacing:3px;margin-top:-8px;'>
  UNIFIED INDUSTRIAL AI PLATFORM · {mod_label.upper()}
</p>""", unsafe_allow_html=True)
with col_s:
    n_done = len(st.session_state.get("results", []))
    st.metric("Analyses run", n_done)

st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []

# ══════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:

    # ── SENSORGUARD ───────────────────────────────────────────────────────────
    if "SensorGuard" in module:
        from utils.data_loader import load_channel, list_channels
        with st.spinner(f"Loading channel {channel}…"):
            data = load_channel(channel)

        prog = st.progress(0, text="Training LSTM Autoencoder…")
        from models.detector import SensorGuardDetector

        det = SensorGuardDetector(window=window, threshold_pct=thr_pct,
                                   hidden=32, epochs=epochs)
        def cb(ep, loss):
            prog.progress(int(ep/epochs*100), text=f"Epoch {ep}/{epochs} · loss={loss:.6f}")
        det.fit(data["train"], progress_cb=cb)
        prog.empty()

        with st.spinner("Scoring…"):
            result  = det.predict(data["test"])
            metrics = det.evaluate(result["predictions"], data["labels"])

        st.session_state.results.append({
            "module": "SensorGuard", "channel": channel, "dataset": dataset,
            "data": data, "result": result, "metrics": metrics,
            "params": {"window":window,"thr_pct":thr_pct,"epochs":epochs,"viz_rng":viz_rng},
        })
        st.rerun()

    # ── INFRAGUARD ────────────────────────────────────────────────────────────
    elif "InfraGuard" in module:
        with st.spinner("Loading SWaT data…"):
            df_tr = pd.read_csv("data/swat/train.csv")
            df_te = pd.read_csv("data/swat/test.csv")
            meta  = json.load(open("data/swat/meta.json"))
            labels_all = df_te["label"].values
            train_arr  = df_tr.values
            test_arr   = df_te.drop("label", axis=1).values

        prog = st.progress(0, text="Training InfraGuard LSTM…")
        from models.infraguard import InfraGuardDetector
        det = InfraGuardDetector(window=window, epochs=epochs,
                                  threshold_pct=thr_pct, batch_size=256, step=4)
        def cb2(ep, loss):
            prog.progress(int(ep/epochs*100), text=f"Epoch {ep}/{epochs} · loss={loss:.6f}")
        det.fit(train_arr, progress_cb=cb2)
        prog.empty()

        with st.spinner("Scoring 51 sensors…"):
            result  = det.predict(test_arr)
            metrics = det.evaluate(result["predictions"], labels_all)

        st.session_state.results.append({
            "module": "InfraGuard", "meta": meta,
            "test": test_arr, "labels": labels_all,
            "result": result, "metrics": metrics,
            "params": {"window":window,"thr_pct":thr_pct,"epochs":epochs,"viz_rng":viz_rng},
            "sensors": meta["sensors"],
        })
        st.rerun()

    # ── FEM BRIDGE ────────────────────────────────────────────────────────────
    else:
        from models.fem_bridge import (generate_bridge_data,
            extract_modal_params_guided, build_stiffness_matrix,
            diagnose_matrix, BRIDGE_PARAMS)

        with st.spinner("Generating bridge sensor data…"):
            data_br, t_arr = generate_bridge_data(t_sec, damage, seed=42)
            modal  = extract_modal_params_guided(data_br, freq_range=(0.05, 2.0))
            K      = build_stiffness_matrix(modal, n_dof=n_modes)
            diag   = diagnose_matrix(K,
                         BRIDGE_PARAMS["freq_nominal"][:n_modes],
                         modal["frequencies_hz"][:n_modes])

        st.session_state.results.append({
            "module": "FEMBridge", "damage": damage,
            "data": data_br, "t": t_arr,
            "modal": modal, "K": K, "diag": diag,
            "params": {"t_sec":t_sec,"n_modes":n_modes,"damage":damage},
        })
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.results:
    st.markdown("""
<div style='text-align:center;padding:80px 40px;'>
  <div style='font-family:"IBM Plex Mono",monospace;font-size:56px;color:#111827;'>⚙</div>
  <div style='font-family:"IBM Plex Mono",monospace;font-size:13px;color:#1A3550;
              letter-spacing:4px;margin-top:8px;'>SELECT A MODULE AND RUN ANALYSIS</div>
  <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;color:#0D1A28;
              letter-spacing:2px;margin-top:16px;line-height:2.5;'>
    SENSORGUARD · NASA SMAP/MSL · LSTM Autoencoder · Anomaly Detection<br>
    INFRAGUARD &nbsp;· SWaT 51 sensors · Water treatment · Attack detection<br>
    FEM BRIDGE &nbsp;· Norwegian bridges · Modal analysis · NOVA-Ω diagnosis
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()

# Afficher le dernier résultat
item = st.session_state.results[-1]

# ── SensorGuard display ───────────────────────────────────────────────────────
if item["module"] == "SensorGuard":
    m   = item["metrics"]
    r   = item["result"]
    d   = item["data"]
    prm = item["params"]
    T   = min(prm["viz_rng"], len(r["scores"]))

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("F1-Score",   f"{m['f1']:.3f}",       delta="≥0.6 ✓" if m['f1']>=0.6 else "<0.6")
    c2.metric("Precision",  f"{m['precision']:.3f}")
    c3.metric("Recall",     f"{m['recall']:.3f}")
    c4.metric("FPR",        f"{m['fpr']:.3f}",       delta="<0.05 ✓" if m['fpr']<0.05 else "high")
    c5.metric("Anomalies",  f"{int(r['predictions'].sum())}")

    st.markdown("---")
    tab1,tab2,tab3 = st.tabs(["Signal & Anomalies","Score Distribution","Report"])

    with tab1:
        t_arr = np.arange(T)
        feat  = d["test"][:T, 0]
        fig   = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7,0.3], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=t_arr, y=feat, mode="lines",
            line=dict(color="#00D4FF",width=1), name="Signal"), row=1, col=1)
        for seq in d.get("anomaly_sequences",[]):
            s,e = seq[0], min(seq[1],T-1)
            if s<T:
                fig.add_vrect(x0=s,x1=e,fillcolor="rgba(255,107,53,0.12)",
                              layer="below",line_width=0,row=1,col=1)
        pred_t = np.where(r["predictions"][:T]==1)[0]
        if len(pred_t):
            fig.add_trace(go.Scatter(x=pred_t, y=feat[pred_t], mode="markers",
                name="Detected", marker=dict(color="#FF6B35",size=4,symbol="x")),row=1,col=1)
        fig.add_trace(go.Scatter(x=t_arr, y=r["scores"][:T], mode="lines",
            line=dict(color="#9D5FFF",width=1.5), fill="tozeroy",
            fillcolor="rgba(157,95,255,0.06)", name="Score"), row=2,col=1)
        fig.add_hline(y=r["threshold"],row=2,col=1,
            line=dict(color="#FF6B35",width=1,dash="dash"),
            annotation_text=f"threshold {r['threshold']:.4f}",
            annotation_font_color="#FF6B35")
        fig.update_layout(**PLOTLY_THEME, height=460,
            legend=dict(bgcolor="#060A12",bordercolor="#1A2535",borderwidth=1))
        fig.update_yaxes(gridcolor="#111827")
        fig.update_xaxes(gridcolor="#111827")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        labels_t = d["labels"][:T]
        sc_norm  = r["scores"][:T][labels_t==0]
        sc_anom  = r["scores"][:T][labels_t==1]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=sc_norm, name="Normal",
            marker_color="#00D4FF", opacity=0.6, nbinsx=50))
        if len(sc_anom):
            fig2.add_trace(go.Histogram(x=sc_anom, name="Anomaly (GT)",
                marker_color="#FF6B35", opacity=0.7, nbinsx=30))
        fig2.add_vline(x=r["threshold"], line=dict(color="#39FF14",width=2,dash="dash"),
            annotation_text="Threshold", annotation_font_color="#39FF14")
        fig2.update_layout(**PLOTLY_THEME, barmode="overlay", height=350,
            title=dict(text="Anomaly score distribution",
                       font=dict(family="IBM Plex Mono",size=12,color="#8BA0BC")))
        st.plotly_chart(fig2, width="stretch")

    with tab3:
        ca,cb_ = st.columns(2)
        with ca:
            st.markdown("**Analysis summary**")
            st.markdown(f"""
<div style='font-family:"IBM Plex Mono",monospace;font-size:11px;color:#8BA0BC;line-height:2.2;'>
DATASET &nbsp;&nbsp;{item['dataset']} · {item['channel']}<br>
POINTS &nbsp;&nbsp;&nbsp;{len(r['scores']):,}<br>
ANOMALIES GT &nbsp;{int(d['labels'].sum()):,} ({100*d['labels'].mean():.1f}%)<br>
DETECTED &nbsp;{int(r['predictions'].sum()):,}<br>
THRESHOLD &nbsp;{r['threshold']:.6f}<br>
WINDOW &nbsp;&nbsp;&nbsp;{prm['window']} pts<br>
EPOCHS &nbsp;&nbsp;&nbsp;{prm['epochs']}
</div>""", unsafe_allow_html=True)
        with cb_:
            st.markdown("**Metrics**")
            st.markdown(f"""
<div style='font-family:"IBM Plex Mono",monospace;font-size:11px;color:#8BA0BC;line-height:2.2;'>
F1-Score &nbsp;&nbsp;{m['f1']:.4f}<br>
Precision &nbsp;{m['precision']:.4f}<br>
Recall &nbsp;&nbsp;&nbsp;{m['recall']:.4f}<br>
FPR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m['fpr']:.4f}<br>
TP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m['TP']:,}<br>
FP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m['FP']:,}<br>
FN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m['FN']:,}
</div>""", unsafe_allow_html=True)

# ── InfraGuard display ────────────────────────────────────────────────────────
elif item["module"] == "InfraGuard":
    m   = item["metrics"]
    r   = item["result"]
    prm = item["params"]
    T   = min(prm["viz_rng"], len(r["scores"]))
    sensors = item["sensors"]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("F1-Score",  f"{m['f1']:.3f}", delta="≥0.7 ✓" if m['f1']>=0.7 else "<0.7")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall",    f"{m['recall']:.3f}")
    c4.metric("FPR",       f"{m['fpr']:.3f}", delta="<0.02 ✓" if m['fpr']<0.02 else "high")
    c5.metric("Sensors",   f"{len(sensors)}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Anomaly Score","Multi-Sensor View","Report"])

    with tab1:
        t_arr = np.arange(T)
        fig   = make_subplots(rows=2,cols=1,shared_xaxes=True,
                               row_heights=[0.6,0.4],vertical_spacing=0.04)
        # Score
        fig.add_trace(go.Scatter(x=t_arr, y=r["scores"][:T], mode="lines",
            line=dict(color="#00D4FF",width=1.2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.05)", name="Score"), row=1,col=1)
        fig.add_hline(y=r["threshold"],row=1,col=1,
            line=dict(color="#FF6B35",width=1,dash="dash"),
            annotation_text=f"threshold {r['threshold']:.4f}",
            annotation_font_color="#FF6B35")
        # Labels vs predictions
        fig.add_trace(go.Scatter(x=t_arr, y=item["labels"][:T], mode="lines",
            line=dict(color="rgba(255,107,53,0.6)",width=1), name="Ground Truth"), row=2,col=1)
        fig.add_trace(go.Scatter(x=t_arr, y=r["predictions"][:T], mode="lines",
            line=dict(color="#39FF14",width=1), name="Predicted"), row=2,col=1)
        fig.update_layout(**PLOTLY_THEME, height=440,
            legend=dict(bgcolor="#060A12",bordercolor="#1A2535",borderwidth=1),
            yaxis2=dict(gridcolor="#111827"))
        st.plotly_chart(fig, width="stretch")

    with tab2:
        n_show = 6
        sel = sensors[:n_show]
        fig3 = make_subplots(rows=n_show, cols=1, shared_xaxes=True,
                              vertical_spacing=0.02,
                              subplot_titles=sel)
        pal  = ["#00D4FF","#9D5FFF","#39FF14","#FF6B35","#FFB800","#FF4080"]
        for i, sname in enumerate(sel):
            j   = sensors.index(sname)
            fig3.add_trace(go.Scatter(x=np.arange(T),
                y=item["test"][:T,j], mode="lines",
                line=dict(color=pal[i],width=0.8), name=sname), row=i+1,col=1)
            fig3.update_yaxes(gridcolor="#111827",row=i+1,col=1)
            fig3.update_xaxes(gridcolor="#111827",row=i+1,col=1)
        fig3.update_layout(**PLOTLY_THEME, height=120*n_show)
        st.plotly_chart(fig3, width="stretch")

    with tab3:
        ca, cb_ = st.columns(2)
        with ca:
            st.markdown("**SWaT Infrastructure**")
            for proc in item["meta"]["processes"]:
                st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;'
                            f'font-size:10px;color:#3A5070;line-height:1.8;">◈ {proc}</div>',
                            unsafe_allow_html=True)
        with cb_:
            st.markdown("**Performance**")
            st.markdown(f"""
<div style='font-family:"IBM Plex Mono",monospace;font-size:11px;color:#8BA0BC;line-height:2.2;'>
F1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m['f1']:.4f}<br>
Precision &nbsp;{m['precision']:.4f}<br>
Recall &nbsp;&nbsp;&nbsp;{m['recall']:.4f}<br>
FPR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m['fpr']:.4f}<br>
TP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m['TP']:,}<br>
FP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m['FP']:,}
</div>""", unsafe_allow_html=True)

# ── FEM Bridge display ────────────────────────────────────────────────────────
else:
    diag   = item["diag"]
    modal  = item["modal"]
    K      = item["K"]
    data_b = item["data"]
    t_arr  = item["t"]
    T      = min(3600, len(t_arr))
    sev    = diag["severity"]
    sev_class = {"NOMINAL":"status-nominal","LOW":"status-nominal",
                 "MEDIUM":"status-medium","HIGH":"status-high",
                 "CRITICAL":"status-critical"}.get(sev,"status-high")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Severity",     sev)
    c2.metric("κ (condition)", f"{diag['kappa']:.2e}")
    c3.metric("Freq drop",    f"{diag['freq_drop_pct']:.1f}%")
    c4.metric("Neg. eigvals", diag["n_neg_eigvals"])

    st.markdown("---")
    tab1,tab2,tab3,tab4 = st.tabs(["Vibration Signal","FFT Spectrum","Stiffness Matrix K","NOVA-Ω Diagnosis"])

    with tab1:
        fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                             vertical_spacing=0.04,
                             subplot_titles=["Vertical acceleration (mid-span)",
                                             "Longitudinal acceleration",
                                             "Strain gauge #1"])
        pal2 = ["#00D4FF","#9D5FFF","#39FF14"]
        for i,(row,ch) in enumerate([(1,0),(2,1),(3,7)]):
            fig.add_trace(go.Scatter(x=t_arr[:T],y=data_b[:T,ch],mode="lines",
                line=dict(color=pal2[i],width=0.8),name=f"ch{ch}"),row=row,col=1)
            fig.update_yaxes(gridcolor="#111827",row=row,col=1)
            fig.update_xaxes(gridcolor="#111827",row=row,col=1)
        fig.update_xaxes(title_text="Time (s)",row=3,col=1)
        fig.update_layout(**PLOTLY_THEME,height=440)
        st.plotly_chart(fig,width="stretch")

    with tab2:
        freq_ax = np.array(modal["freq_axis"])
        fft_sp  = np.array(modal["fft_spectrum"])
        mask    = freq_ax <= 2.5
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=freq_ax[mask],y=fft_sp[mask],
            mode="lines",line=dict(color="#9D5FFF",width=1.2),
            fill="tozeroy",fillcolor="rgba(157,95,255,0.08)",name="FFT"))
        for f in modal["frequencies_hz"]:
            if f <= 2.5:
                fig2.add_vline(x=f,line=dict(color="#00D4FF",width=1,dash="dot"),
                    annotation_text=f"{f:.3f} Hz",
                    annotation_font=dict(color="#00D4FF",size=9))
        fig2.update_layout(**PLOTLY_THEME, height=340,
            title=dict(text="Modal FFT spectrum — extracted frequencies",
                       font=dict(family="IBM Plex Mono",size=12,color="#8BA0BC")))
        fig2.update_xaxes(title_text="Frequency (Hz)", gridcolor="#111827")
        fig2.update_yaxes(title_text="Amplitude", gridcolor="#111827")
        st.plotly_chart(fig2,width="stretch")

        st.markdown("**Identified modal frequencies**")
        cols = st.columns(len(modal["frequencies_hz"]))
        for i,(f,col) in enumerate(zip(modal["frequencies_hz"],cols)):
            col.metric(f"Mode {i+1}", f"{f:.4f} Hz")

    with tab3:
        n = K.shape[0]
        fig3 = go.Figure(data=go.Heatmap(
            z=K, colorscale="Blues",
            colorbar=dict(tickfont=dict(family="IBM Plex Mono",size=9,color="#8BA0BC")),
            hovertemplate="K[%{y},%{x}] = %{z:.2e}<extra></extra>",
        ))
        fig3.update_layout(**PLOTLY_THEME,height=380,
            title=dict(text=f"Stiffness matrix K ({n}×{n} DOF) — from measured modal data",
                       font=dict(family="IBM Plex Mono",size=12,color="#8BA0BC")),
            xaxis=dict(title="DOF"),yaxis=dict(title="DOF"))
        st.plotly_chart(fig3,width="stretch")

        ca,cb_ = st.columns(2)
        with ca:
            st.markdown("**Eigenvalues λ (N/m)**")
            eigvals = diag["eigvals"]
            fig_eig = go.Figure(go.Bar(
                y=eigvals, marker_color=["#FF2020" if v<0 else "#00D4FF" for v in eigvals],
                name="Eigenvalues"))
            fig_eig.update_layout(**PLOTLY_THEME, height=220)
            fig_eig.update_xaxes(title_text="Mode", gridcolor="#111827")
            fig_eig.update_yaxes(title_text="λ", gridcolor="#111827")
            st.plotly_chart(fig_eig,width="stretch")
        with cb_:
            st.markdown("**Matrix properties**")
            st.markdown(f"""
<div style='font-family:"IBM Plex Mono",monospace;font-size:11px;color:#8BA0BC;line-height:2.4;'>
Condition κ &nbsp;&nbsp;{diag['kappa']:.3e}<br>
Min eigval &nbsp;&nbsp;{diag['min_eigval']:.3e}<br>
Max eigval &nbsp;&nbsp;{diag['max_eigval']:.3e}<br>
Neg eigvals &nbsp;{diag['n_neg_eigvals']}<br>
Sym error &nbsp;&nbsp;{diag['sym_error']:.6f}<br>
Diag dominant {diag['diag_dominant']}
</div>""", unsafe_allow_html=True)

    with tab4:
        sev_col = {"NOMINAL":"#39FF14","LOW":"#00D4FF","MEDIUM":"#FFB800",
                   "HIGH":"#FF6B35","CRITICAL":"#FF2020"}.get(sev,"#FF6B35")
        st.markdown(f"""
<div style='border:2px solid {sev_col};padding:16px 20px;margin-bottom:16px;'>
  <span style='font-family:"IBM Plex Mono",monospace;font-size:11px;
               color:{sev_col};letter-spacing:3px;'>NOVA-Ω SEVERITY : {sev}</span>
</div>""", unsafe_allow_html=True)

        for d_item in diag["diagnostics"]:
            s = d_item["severity"]
            st.markdown(f"""
<div class='diag-box-{s}'>
  <b>{d_item['pattern']}</b> [{s}]<br>
  {d_item['description']}<br>
  <span style='opacity:0.7'>Cause : {d_item['cause']}</span><br>
  <span style='opacity:0.7'>Action: {d_item['action']}</span>
</div>""", unsafe_allow_html=True)

        # Comparaison fréquences nominal vs mesuré
        st.markdown("<br>**Frequency comparison : nominal vs measured**", unsafe_allow_html=True)
        from models.fem_bridge import BRIDGE_PARAMS
        nom = BRIDGE_PARAMS["freq_nominal"][:len(modal["frequencies_hz"])]
        mea = modal["frequencies_hz"]
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name="Nominal", x=[f"Mode {i+1}" for i in range(len(nom))],
            y=nom, marker_color="rgba(0,212,255,0.3)",
            marker_line=dict(color="#00D4FF",width=1)))
        fig5.add_trace(go.Bar(name="Measured", x=[f"Mode {i+1}" for i in range(len(mea))],
            y=mea, marker_color="rgba(255,107,53,0.6)",
            marker_line=dict(color="#FF6B35",width=1)))
        fig5.update_layout(**PLOTLY_THEME, barmode="group", height=280,
            legend=dict(bgcolor="#060A12",bordercolor="#1A2535"))
        fig5.update_xaxes(gridcolor="#111827")
        fig5.update_yaxes(title_text="Frequency (Hz)", gridcolor="#111827")
        st.plotly_chart(fig5, width="stretch")

# ── History ───────────────────────────────────────────────────────────────────
if len(st.session_state.results) > 1:
    st.markdown("---")
    st.markdown("### ANALYSIS HISTORY")
    for i, r_item in enumerate(reversed(st.session_state.results)):
        mod = r_item["module"]
        mono = "IBM Plex Mono"
        style = f"font-family:'{mono}',monospace;font-size:10px;color:#3A5070;"
        if mod == "SensorGuard":
            txt = f"#{i+1} SensorGuard · {r_item['channel']} · F1={r_item['metrics']['f1']:.3f}"
        elif mod == "InfraGuard":
            txt = f"#{i+1} InfraGuard · SWaT 51 sensors · F1={r_item['metrics']['f1']:.3f}"
        else:
            txt = f"#{i+1} FEM Bridge · damage={r_item['damage']:.2f} · {r_item['diag']['severity']}"
        st.markdown(f'<span style="{style}">{txt}</span>', unsafe_allow_html=True)

if len(st.session_state.results) > 0:
    if st.button("Clear all", key="clear_all"):
        st.session_state.results = []
        st.rerun()
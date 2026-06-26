import pickle
import os

import altair as alt
import av
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from classifier import CLASSES, ASSETS_DIR
from pipeline import (
    TwoStagePipeline,
    apply_overlay,
    draw_bbox_on_pil,
    draw_no_banana_pil,
)

st.set_page_config(
    page_title="Banana Ripeness Classifier",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

YOLO_CONF = 0.40
UNCERTAIN_THRESHOLD = 0.75

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

COLORS = {
    "unripe":   "#2D6A4F",
    "ripe":     "#D4A017",
    "overripe": "#A0522D",
    "rotten":   "#4A1A0A",
}

COLORS_LIGHT = {
    "unripe":   "#E8F5E9",
    "ripe":     "#FFF9E6",
    "overripe": "#FBE9D0",
    "rotten":   "#F5E6E0",
}

GUIDE = {
    "unripe": {
        "title": "Unripe",
        "emoji": "🟢",
        "tag": "Not ready yet",
        "desc": "Firm, green and starchy. Leave at room temperature for a few days to sweeten before eating.",
    },
    "ripe": {
        "title": "Ripe",
        "emoji": "🍌",
        "tag": "Eat now",
        "desc": "Sweet, soft and bright yellow. Perfect for eating fresh, in cereal, smoothies or fruit salads.",
    },
    "overripe": {
        "title": "Overripe",
        "emoji": "🟡",
        "tag": "Best for baking",
        "desc": "Very sweet and soft with brown speckles. Ideal for banana bread, pancakes and smoothies.",
    },
    "rotten": {
        "title": "Rotten",
        "emoji": "🟤",
        "tag": "Discard",
        "desc": "Mushy, fermented, leaking or moldy. Past its prime — throw it out or compost it.",
    },
}

PROCESS_EVERY_N_FRAMES = 3


@st.cache_resource(show_spinner="Loading models…")
def load_pipeline() -> TwoStagePipeline:
    return TwoStagePipeline(yolo_conf=YOLO_CONF)


@st.cache_data
def get_history():
    path = os.path.join(ASSETS_DIR, "training_history.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


class BananaVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._pipeline   = load_pipeline()
        self.last_result = None
        self._last_frame = None
        self._frame_i    = 0
        self._lock       = __import__("threading").Lock()

    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        self._frame_i += 1
        if self._frame_i % PROCESS_EVERY_N_FRAMES == 0:
            annotated, result = apply_overlay(
                bgr, self._pipeline.detector, self._pipeline.classifier, COLORS,
            )
            with self._lock:
                self._last_frame = annotated
                self.last_result = result
        else:
            with self._lock:
                annotated = self._last_frame
        if annotated is None:
            annotated = bgr
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def inject_css():
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            /* ── Full page background takeover ── */
            html, body, .stApp, [data-testid="stAppViewContainer"] {
                background-color: #FAFAF7 !important;
                font-family: 'Inter', sans-serif !important;
            }
            [data-testid="stHeader"] {
                background-color: #FAFAF7 !important;
            }
            [data-testid="stSidebar"] {
                background-color: #F0EFE8 !important;
            }
            .block-container {
                padding-top: 0 !important;
                padding-bottom: 4rem !important;
                max-width: 1200px !important;
            }

            /* ── Hero banner ── */
            .hero-wrap {
                background: linear-gradient(160deg, #1A3A2A 0%, #2D6A4F 60%, #3D8B5E 100%);
                border-radius: 0 0 32px 32px;
                padding: 52px 48px 44px;
                margin: -4rem -4rem 2.5rem -4rem;
                position: relative;
                overflow: hidden;
            }
            .hero-wrap::before {
                content: "🍌";
                position: absolute;
                right: 48px;
                top: 50%;
                transform: translateY(-50%);
                font-size: 120px;
                opacity: 0.15;
                filter: grayscale(0.3);
            }
            .hero-eyebrow {
                display: inline-block;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.2);
                color: #A8D5B5;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                padding: 6px 14px;
                border-radius: 999px;
                margin-bottom: 18px;
            }
            .hero-title {
                font-family: 'Playfair Display', serif !important;
                font-size: clamp(2.2rem, 5vw, 3.6rem);
                font-weight: 900;
                color: #FFFFFF;
                line-height: 1.05;
                margin: 0 0 14px;
                letter-spacing: -0.02em;
            }
            .hero-sub {
                font-size: 1rem;
                color: #B8D4C0;
                max-width: 580px;
                line-height: 1.65;
                font-weight: 400;
            }

            /* ── Stat row ── */
            .stat-row {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 14px;
                margin-bottom: 2rem;
            }
            .stat-card {
                background: #FFFFFF;
                border: 1px solid #E8E6DE;
                border-radius: 16px;
                padding: 20px 22px;
                box-shadow: 0 2px 12px rgba(30,50,35,0.06);
            }
            .stat-card .sv {
                font-family: 'Playfair Display', serif;
                font-size: 1.8rem;
                font-weight: 700;
                color: #1A3A2A;
                line-height: 1;
            }
            .stat-card .sl {
                font-size: 0.72rem;
                font-weight: 700;
                color: #8A8A7A;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                margin-top: 5px;
            }

            /* ── Section headings ── */
            .sec-title {
                font-family: 'Playfair Display', serif;
                font-size: 1.6rem;
                font-weight: 700;
                color: #1A3A2A;
                margin: 0 0 4px;
            }
            .sec-sub {
                font-size: 0.9rem;
                color: #7A7A68;
                font-weight: 500;
                margin-bottom: 18px;
            }

            /* ── Result card ── */
            .result-card {
                border-radius: 20px;
                padding: 26px 28px;
                color: #fff;
                box-shadow: 0 12px 32px -16px rgba(0,0,0,0.35);
                margin-bottom: 4px;
            }
            .result-card .rc-eyebrow {
                font-size: 0.72rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                opacity: 0.8;
                margin-bottom: 8px;
            }
            .result-card .rc-class {
                font-family: 'Playfair Display', serif;
                font-size: 2.6rem;
                font-weight: 900;
                line-height: 1;
                margin-bottom: 6px;
            }
            .result-card .rc-conf {
                font-size: 1.05rem;
                font-weight: 600;
                opacity: 0.92;
            }
            .result-card .rc-tag {
                display: inline-block;
                margin-top: 14px;
                background: rgba(255,255,255,0.2);
                border: 1px solid rgba(255,255,255,0.3);
                padding: 5px 14px;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 700;
            }

            /* ── No banana panel ── */
            .no-banana {
                background: #FFF5F5;
                border: 2px dashed #E0AAAA;
                border-radius: 20px;
                padding: 32px;
                text-align: center;
            }
            .no-banana .nb-title {
                font-family: 'Playfair Display', serif;
                font-size: 1.7rem;
                font-weight: 700;
                color: #8B3A3A;
                margin-bottom: 8px;
            }
            .no-banana .nb-sub {
                font-size: 0.92rem;
                color: #A05A5A;
                font-weight: 500;
            }

            /* ── Guide cards ── */
            .guide-card {
                background: #FFFFFF;
                border: 1px solid #E8E6DE;
                border-top: 5px solid var(--gc);
                border-radius: 16px;
                padding: 20px;
                height: 100%;
                box-shadow: 0 2px 12px rgba(30,50,35,0.05);
            }
            .guide-card .gc-emoji { font-size: 1.8rem; margin-bottom: 8px; }
            .guide-card .gc-tag {
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--gc);
                margin-bottom: 4px;
            }
            .guide-card .gc-title {
                font-family: 'Playfair Display', serif;
                font-size: 1.2rem;
                font-weight: 700;
                color: #1A3A2A;
                margin-bottom: 10px;
            }
            .guide-card .gc-desc {
                font-size: 0.88rem;
                color: #5A5A4A;
                line-height: 1.6;
                font-weight: 400;
            }

            /* ── Uncertainty cards ── */
            .unc-card {
                background: #FFFFFF;
                border: 1px solid #E8E6DE;
                border-top: 5px solid var(--uc);
                border-radius: 14px;
                padding: 18px;
                text-align: center;
            }
            .unc-card .uc-name {
                font-family: 'Playfair Display', serif;
                font-size: 1.2rem;
                font-weight: 700;
                color: #1A3A2A;
            }
            .unc-card .uc-pct {
                font-size: 1.6rem;
                font-weight: 800;
                color: var(--uc);
                margin: 4px 0;
            }
            .unc-card .uc-tag {
                font-size: 0.78rem;
                font-weight: 600;
                color: #8A8A7A;
            }

            /* ── Pipeline badge ── */
            .pipe-badge {
                display: inline-block;
                background: #EAF3EC;
                border: 1px solid #C2DBC8;
                color: #2D6A4F;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-bottom: 16px;
            }

            /* ── Tabs ── */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                background: #EEEEE6;
                padding: 6px;
                border-radius: 14px;
            }
            .stTabs [data-baseweb="tab"] {
                font-weight: 600;
                font-size: 0.92rem;
                padding: 8px 18px;
                border-radius: 10px;
                color: #5A5A4A;
            }
            .stTabs [aria-selected="true"] {
                background: #FFFFFF !important;
                color: #1A3A2A !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            [data-testid="stTabPanel"] {
                padding-top: 1.5rem;
            }

            /* ── Footer ── */
            .footer {
                text-align: center;
                color: #AAAAAA;
                font-size: 0.78rem;
                margin-top: 3rem;
                padding-top: 1.5rem;
                border-top: 1px solid #E8E6DE;
            }

            /* ── Responsive ── */
            @media (max-width: 768px) {
                .hero-wrap { padding: 36px 24px 32px; margin: -2rem -1rem 2rem; }
                .hero-wrap::before { display: none; }
                .stat-row { grid-template-columns: repeat(2, 1fr); }
            }
        </style>
    """, unsafe_allow_html=True)


def confidence_chart(probs: dict):
    order = [c.capitalize() for c in CLASSES]
    df = pd.DataFrame([
        {"Class": c.capitalize(), "key": c, "Probability": probs[c]}
        for c in CLASSES
    ])
    df["pct"] = (df["Probability"] * 100).round(1).astype(str) + "%"
    base = alt.Chart(df)
    bars = base.mark_bar(
        cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=28
    ).encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title=None, grid=True,
                              tickCount=5, labelColor="#8A8A7A", gridColor="#EBEBEB")),
        y=alt.Y("Class:N", sort=order, title=None,
                axis=alt.Axis(labelFontSize=13, labelFontWeight="bold",
                              labelColor="#1A3A2A")),
        color=alt.Color("key:N",
                        scale=alt.Scale(domain=CLASSES, range=[COLORS[c] for c in CLASSES]),
                        legend=None),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".1%")],
    )
    text = base.mark_text(
        align="left", dx=6, fontWeight="bold", fontSize=12, color="#1A3A2A"
    ).encode(
        x="Probability:Q",
        y=alt.Y("Class:N", sort=order),
        text="pct:N",
    )
    return (bars + text).properties(height=200).configure_view(
        strokeOpacity=0, fill="#FAFAF7"
    ).configure(background="#FAFAF7")


def history_chart(values_train, values_val, title, fmt):
    n = len(values_train)
    df = pd.DataFrame({
        "Epoch": list(range(1, n + 1)) * 2,
        "Value": list(values_train) + list(values_val),
        "Set": ["Training"] * n + ["Validation"] * n,
    })
    return alt.Chart(df).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1, title="Epoch",
                                          labelColor="#8A8A7A")),
        y=alt.Y("Value:Q", axis=alt.Axis(format=fmt, title=None,
                                          labelColor="#8A8A7A")),
        color=alt.Color("Set:N",
                        scale=alt.Scale(domain=["Training", "Validation"],
                                        range=["#2D6A4F", "#D4A017"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Epoch", alt.Tooltip("Value:Q", format=".3f"), "Set"],
    ).properties(height=260, title=alt.TitleParams(
        title, color="#1A3A2A", fontSize=14, fontWeight="bold"
    )).configure_view(strokeOpacity=0, fill="#FAFAF7").configure(background="#FAFAF7")


def render_header():
    st.markdown("""
        <div class="hero-wrap">
            <div class="hero-eyebrow">● 98.22 % test accuracy &nbsp;·&nbsp; 13,000+ images &nbsp;·&nbsp; MobileNetV2</div>
            <div class="hero-title">Banana Ripeness<br>Classifier</div>
            <div class="hero-sub">Point your camera at a banana — YOLOv8 locates it instantly,
            then a fine-tuned MobileNetV2 classifies its ripeness stage with a live confidence breakdown.</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="stat-row">
            <div class="stat-card"><div class="sv">98.22%</div><div class="sl">Test accuracy</div></div>
            <div class="stat-card"><div class="sv">13,000+</div><div class="sl">Training images</div></div>
            <div class="stat-card"><div class="sv">4</div><div class="sl">Ripeness classes</div></div>
            <div class="stat-card"><div class="sv">MobileNetV2</div><div class="sl">Architecture</div></div>
        </div>
    """, unsafe_allow_html=True)
    
def render_confusion_matrix():
    path = os.path.join(
        ASSETS_DIR, 
        "Confusion_Matrix_BC.png"
    )
    if not os.path.exists(path):
        st.warning("Confusion matrix not found.")
        return
    img=Image.open(path)
    st.image(
        img, 
        use_container_width=True
    )

def render_sample_predictions():
    path = os.path.join(
        ASSETS_DIR, 
        "Sample_Predictions_BC.png"
    )
    if not os.path.exists(path):
        st.warning("Confusion matrix not found.")
        return
    img=Image.open(path)
    st.image(
        img, 
        use_container_width=True
    )

def render_uncertainty_warning(res: dict):
    if not res.get("uncertain"):
        return
    conf = res["top_conf"]
    c1, c2 = res["ordered"][0], res["ordered"][1]
    st.warning(
        f"⚠️ **Uncertain** — confidence is {conf*100:.1f}% (below 75%). "
        f"Could be **{GUIDE[c1[0]]['title']}** or **{GUIDE[c2[0]]['title']}**."
    )
    col1, col2 = st.columns(2)
    for col, (cls, p) in zip((col1, col2), (c1, c2)):
        with col:
            st.markdown(f"""
                <div class="unc-card" style="--uc:{COLORS[cls]};">
                    <div class="uc-name">{GUIDE[cls]['title']}</div>
                    <div class="uc-pct">{p*100:.1f}%</div>
                    <div class="uc-tag">{GUIDE[cls]['tag']}</div>
                </div>
            """, unsafe_allow_html=True)


def render_ripeness_panel(res: dict, show_chart: bool = True):
    top  = res["top_class"]
    conf = res["top_conf"]
    st.markdown(f"""
        <div class="result-card" style="background:{COLORS[top]};">
            <div class="rc-eyebrow">Prediction</div>
            <div class="rc-class">{GUIDE[top]['emoji']} {GUIDE[top]['title']}</div>
            <div class="rc-conf">{conf*100:.1f}% confidence</div>
            <div class="rc-tag">{GUIDE[top]['tag']}</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(
        f"<p style='margin-top:12px;color:#5A5A4A;font-size:0.92rem;line-height:1.6;'>"
        f"{GUIDE[top]['desc']}</p>",
        unsafe_allow_html=True,
    )
    render_uncertainty_warning(res)
    if show_chart:
        st.markdown(
            "<p style='font-weight:700;color:#1A3A2A;font-size:0.85rem;"
            "text-transform:uppercase;letter-spacing:0.06em;margin-top:16px;"
            "margin-bottom:4px;'>Confidence breakdown</p>",
            unsafe_allow_html=True,
        )
        st.altair_chart(confidence_chart(res["probs"]), use_container_width=True)


def render_no_banana_panel():
    st.markdown("""
        <div class="no-banana">
            <div class="nb-title">🚫 No banana detected</div>
            <div class="nb-sub">YOLOv8 found no banana with ≥40% confidence.<br>
            Point the camera at a clear, well-lit banana and hold it steady.</div>
        </div>
    """, unsafe_allow_html=True)


def render_upload_result(image: Image.Image, pipeline_result: dict):
    if not pipeline_result["banana_detected"]:
        annotated = draw_no_banana_pil(image)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.image(annotated, use_container_width=True)
        with c2:
            render_no_banana_panel()
        return
    det      = pipeline_result["detection"]
    res      = pipeline_result["ripeness"]
    annotated = draw_bbox_on_pil(image, det, res, COLORS)
    c1, c2   = st.columns([1, 1], gap="large")
    with c1:
        st.image(annotated, use_container_width=True)
    with c2:
        st.caption(f"Detection confidence: {det['conf']*100:.1f}%")
        render_ripeness_panel(res)


def render_guide():
    st.markdown('<div class="sec-title">Ripeness guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">What each stage means in practice.</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, cls in zip(cols, CLASSES):
        g = GUIDE[cls]
        with col:
            st.markdown(f"""
                <div class="guide-card" style="--gc:{COLORS[cls]};">
                    <div class="gc-emoji">{g['emoji']}</div>
                    <div class="gc-tag">{g['tag']}</div>
                    <div class="gc-title">{g['title']}</div>
                    <div class="gc-desc">{g['desc']}</div>
                </div>
            """, unsafe_allow_html=True)


def render_history():
    st.markdown('<div class="sec-title">Training history</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Accuracy and loss across epochs — training vs. validation.</div>',
        unsafe_allow_html=True,
    )

    try:
        h = get_history()
        m1, m2, m3 = st.columns(3, gap='small')
        m1.metric("Epochs trained",       len(h["accuracy"]))
        m2.metric("Final train accuracy", f"{h['accuracy'][-1]*100:.2f}%")
        m3.metric("Final val accuracy",   f"{h['val_accuracy'][-1]*100:.2f}%")
    except Exception:
        pass

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    hist_path = os.path.join(ASSETS_DIR, "Training_History_BC.png")
    if os.path.exists(hist_path):
        st.image(Image.open(hist_path), use_container_width=True)
    else:
        st.warning("Training_History_BC.png not found in assets/")

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-title" style="font-size:1.3rem;">Confusion matrix</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">Per-class prediction breakdown on the held-out test set.</div>',
        unsafe_allow_html=True,
    )
    render_confusion_matrix()

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-title" style="font-size:1.3rem;">Sample predictions</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">Nine randomly selected test images — green title = correct, red = incorrect.</div>',
        unsafe_allow_html=True,
    )
    render_sample_predictions()


@st.fragment(run_every=1.0)
def render_live_feed_stats(webrtc_ctx):
    if not webrtc_ctx.state.playing:
        return
    processor = webrtc_ctx.video_processor
    if processor is None or processor.last_result is None:
        st.caption("Waiting for first detection…")
        return
    result = processor.last_result
    if not result["banana_detected"]:
        st.info("No banana in frame — hold a banana steady in front of the camera.")
        return
    res = result["ripeness"]
    det = result["detection"]
    st.caption(f"YOLO detection confidence: {det['conf']*100:.1f}%")
    render_ripeness_panel(res, show_chart=True)


def render_camera_tab():
    st.markdown('<div class="sec-title">Live camera</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Real-time two-stage pipeline running in your browser.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipe-badge">Stage 1: YOLOv8n (COCO class 46) &nbsp;→&nbsp; Stage 2: MobileNetV2 ripeness</div>',
        unsafe_allow_html=True,
    )
    load_pipeline()
    webrtc_ctx = webrtc_streamer(
        key="banana-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=BananaVideoProcessor,
        async_processing=True,
    )
    if webrtc_ctx.state.playing:
        render_live_feed_stats(webrtc_ctx)
    else:
        st.info("▶️ Press **Start** to open the live camera feed.")


def main():
    inject_css()
    render_header()

    tab_cam, tab_up, tab_guide, tab_hist = st.tabs([
        "📷  Live camera",
        "🖼️  Upload photo",
        "📖  Ripeness guide",
        "📈  Training history",
    ])

    with tab_cam:
        render_camera_tab()

    with tab_up:
        st.markdown('<div class="sec-title">Upload a photo</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-sub">YOLOv8 locates the banana, then MobileNetV2 classifies its ripeness.</div>',
            unsafe_allow_html=True,
        )
        up = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "webp"],
            key="up", label_visibility="collapsed",
        )
        if up is not None:
            img = Image.open(up)
            with st.spinner("Analysing…"):
                result = load_pipeline().process_pil(img)
            render_upload_result(img, result)
        else:
            st.info("⬆️ Drag and drop or browse to upload a banana photo.")

    with tab_guide:
        render_guide()

    with tab_hist:
        render_history()

    st.markdown(
        '<div class="footer">YOLOv8n + MobileNetV2 &nbsp;·&nbsp; 4-class banana ripeness &nbsp;·&nbsp; streamlit-webrtc</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
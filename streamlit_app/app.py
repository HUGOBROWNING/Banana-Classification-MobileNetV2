"""Banana Ripeness Classifier - Streamlit app.

Live camera (streamlit-webrtc + YOLOv8n) and image upload powered by a
two-stage pipeline: banana detection → MobileNetV2 ripeness classification.
"""

import pickle
import os

import altair as alt
import av
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from classifier import CLASSES, ASSETS_DIR
from pipeline import (
    TwoStagePipeline,
    apply_overlay,
    draw_bbox_on_pil,
    draw_no_banana_pil,
)

# --------------------------------------------------------------------------- #
# Config & constants
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Banana Ripeness Classifier",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

YOLO_CONF = 0.40
UNCERTAIN_THRESHOLD = 0.75

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

COLORS = {
    "unripe": "#4C9A2A",
    "ripe": "#F2B705",
    "overripe": "#C07A1E",
    "rotten": "#6E3B2A",
}

GUIDE = {
    "unripe": {
        "title": "Unripe",
        "tag": "Not ready yet",
        "desc": "Firm, green and starchy. Leave at room temperature for a few days to sweeten before eating.",
    },
    "ripe": {
        "title": "Ripe",
        "tag": "Eat now",
        "desc": "Sweet, soft and bright yellow. Perfect for eating fresh, in cereal, smoothies or fruit salads.",
    },
    "overripe": {
        "title": "Overripe",
        "tag": "Best for baking",
        "desc": "Very sweet and soft with brown speckles. Ideal for banana bread, pancakes and smoothies.",
    },
    "rotten": {
        "title": "Rotten",
        "tag": "Discard",
        "desc": "Mushy, fermented, leaking or moldy. Past its prime — throw it out or compost it.",
    },
}

FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
PROCESS_EVERY_N_FRAMES = 3


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading YOLO + ripeness models…")
def load_pipeline() -> TwoStagePipeline:
    return TwoStagePipeline(yolo_conf=YOLO_CONF)


@st.cache_data
def get_history():
    path = os.path.join(ASSETS_DIR, "training_history.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# WebRTC video processor
# --------------------------------------------------------------------------- #
class BananaVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._pipeline    = load_pipeline()
        self.last_result  = None
        self._last_frame  = None
        self._frame_i     = 0
        self._lock        = __import__("threading").Lock()

    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        self._frame_i += 1

        if self._frame_i % PROCESS_EVERY_N_FRAMES == 0:
            annotated, result = apply_overlay(
                bgr,
                self._pipeline.detector,
                self._pipeline.classifier,
                COLORS,
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


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"], .stApp { font-family: 'Manrope', sans-serif; }
            h1, h2, h3, .hero-title { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }
            .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
            .hero {
                background: linear-gradient(135deg, #FFF7DC 0%, #FDEFC0 55%, #FBE3A0 100%);
                border: 1px solid #F2DfA0; border-radius: 26px; padding: 38px 40px;
                box-shadow: 0 18px 40px -24px rgba(180, 140, 20, 0.55);
            }
            .hero-title { font-size: clamp(2rem, 4.5vw, 3.2rem); font-weight: 700; color: #1F1B12; margin: 4px 0 6px; line-height: 1.05; }
            .hero-sub { font-size: 1.02rem; color: #6B5F3E; max-width: 640px; font-weight: 500; }
            .hero-badge {
                display: inline-flex; align-items: center; gap: 8px;
                background: #1F1B12; color: #FFD23F; font-weight: 700; font-size: .8rem;
                padding: 7px 14px; border-radius: 999px; letter-spacing: .04em; text-transform: uppercase;
            }
            .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 18px; }
            .stat {
                background: #FFFFFF; border: 1px solid #EFE7D2; border-radius: 18px; padding: 18px 20px;
                box-shadow: 0 10px 26px -22px rgba(80,60,10,.6);
            }
            .stat .v { font-family: 'Fraunces', serif; font-size: 1.7rem; font-weight: 700; color: #1F1B12; }
            .stat .l { font-size: .8rem; color: #8A7C58; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; margin-top: 2px; }
            .result-card {
                border-radius: 20px; padding: 24px 26px; color: #fff;
                box-shadow: 0 16px 34px -20px rgba(0,0,0,.45);
            }
            .result-card .label { font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; opacity: .9; }
            .result-card .cls { font-family: 'Fraunces', serif; font-size: 2.4rem; font-weight: 700; line-height: 1; margin: 6px 0 4px; }
            .result-card .conf { font-size: 1.05rem; font-weight: 600; opacity: .95; }
            .result-card .tag { display:inline-block; margin-top: 12px; background: rgba(255,255,255,.22); padding: 6px 12px; border-radius: 999px; font-weight: 700; font-size: .85rem; }
            .nobanana {
                background: #FBEFEF; border: 1.5px dashed #D98A8A; color: #9A3B3B;
                border-radius: 20px; padding: 28px; text-align: center;
            }
            .nobanana .big { font-family: 'Fraunces', serif; font-size: 1.9rem; font-weight: 700; }
            .guide-card {
                background:#fff; border:1px solid #EFE7D2; border-left: 6px solid var(--c);
                border-radius: 16px; padding: 18px 20px; height: 100%;
                box-shadow: 0 10px 24px -22px rgba(80,60,10,.6);
            }
            .guide-card .gt { font-family:'Fraunces',serif; font-size:1.25rem; font-weight:700; color:#1F1B12; }
            .guide-card .gtag { font-size:.78rem; font-weight:700; color: var(--c); text-transform:uppercase; letter-spacing:.05em; }
            .guide-card .gd { font-size:.92rem; color:#6B5F3E; margin-top:8px; font-weight:500; line-height:1.5; }
            .top2-card {
                background:#fff; border:1px solid #EFE7D2; border-top: 5px solid var(--c);
                border-radius:14px; padding:16px 18px; text-align:center;
            }
            .top2-card .n { font-family:'Fraunces',serif; font-size:1.3rem; font-weight:700; color:#1F1B12; }
            .top2-card .p { font-size:1.4rem; font-weight:800; color:var(--c); }
            .section-h { font-family:'Fraunces',serif; font-size:1.7rem; font-weight:700; color:#1F1B12; margin: 6px 0 2px; }
            .section-sub { color:#8A7C58; font-weight:500; margin-bottom: 14px; }
            .pipeline-badge {
                display:inline-block; background:#FFF7DC; border:1px solid #F2DfA0;
                border-radius:10px; padding:6px 12px; font-size:.82rem; font-weight:600; color:#6B5F3E; margin-bottom:12px;
            }
            .stTabs [data-baseweb="tab-list"] { gap: 6px; }
            .stTabs [data-baseweb="tab"] { font-weight:700; font-size:.98rem; padding: 8px 16px; border-radius: 12px 12px 0 0; }
            @media (max-width: 760px) {
                .stat-grid { grid-template-columns: repeat(2, 1fr); }
                .hero { padding: 26px 22px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def confidence_chart(probs: dict):
    order = [c.capitalize() for c in CLASSES]
    df = pd.DataFrame(
        [{"Class": c.capitalize(), "key": c, "Probability": probs[c]} for c in CLASSES]
    )
    df["pct"] = (df["Probability"] * 100).round(1).astype(str) + "%"
    base = alt.Chart(df)
    bars = base.mark_bar(cornerRadiusTopRight=7, cornerRadiusBottomRight=7, height=30).encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title=None, grid=True, tickCount=5)),
        y=alt.Y("Class:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, labelFontWeight="bold")),
        color=alt.Color("key:N", scale=alt.Scale(domain=CLASSES, range=[COLORS[c] for c in CLASSES]), legend=None),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".1%")],
    )
    text = base.mark_text(align="left", dx=6, fontWeight="bold", fontSize=12, color="#2A271F").encode(
        x="Probability:Q", y=alt.Y("Class:N", sort=order), text="pct:N",
    )
    return (bars + text).properties(height=210).configure_view(strokeOpacity=0)


def history_chart(values_train, values_val, title, fmt):
    n = len(values_train)
    epochs = list(range(1, n + 1))
    df = pd.DataFrame({
        "Epoch": epochs * 2,
        "Value": list(values_train) + list(values_val),
        "Set": ["Training"] * n + ["Validation"] * n,
    })
    return alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1, title="Epoch")),
        y=alt.Y("Value:Q", axis=alt.Axis(format=fmt, title=None)),
        color=alt.Color("Set:N", scale=alt.Scale(domain=["Training", "Validation"], range=["#E29400", "#4C9A2A"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Epoch", alt.Tooltip("Value:Q", format=".3f"), "Set"],
    ).properties(height=280, title=title).configure_view(strokeOpacity=0)


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_header():
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">● 98.22% test accuracy</span>
            <div class="hero-title">Banana Ripeness Classifier</div>
            <div class="hero-sub">Point your camera at a banana — YOLOv8n finds it, then a MobileNetV2 model
            trained on 13,000+ images classifies ripeness in real time with a live confidence breakdown.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="stat-grid">
            <div class="stat"><div class="v">98.22%</div><div class="l">Test accuracy</div></div>
            <div class="stat"><div class="v">13,000+</div><div class="l">Training images</div></div>
            <div class="stat"><div class="v">4</div><div class="l">Ripeness classes</div></div>
            <div class="stat"><div class="v">MobileNetV2</div><div class="l">Architecture</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_uncertainty_warning(res: dict):
    if not res.get("uncertain"):
        return
    conf = res["top_conf"]
    c1, c2 = res["ordered"][0], res["ordered"][1]
    st.warning(
        f"⚠️ **Uncertain** — top confidence is {conf*100:.1f}% (below 75%). "
        f"Could be **{GUIDE[c1[0]]['title']}** or **{GUIDE[c2[0]]['title']}**:"
    )
    cc1, cc2 = st.columns(2)
    for col, (cls, p) in zip((cc1, cc2), (c1, c2)):
        with col:
            st.markdown(
                f"""
                <div class="top2-card" style="--c:{COLORS[cls]};">
                    <div class="n">{GUIDE[cls]['title']}</div>
                    <div class="p">{p*100:.1f}%</div>
                    <div style="color:#8A7C58;font-size:.85rem;font-weight:600;margin-top:4px;">{GUIDE[cls]['tag']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_ripeness_panel(res: dict, show_chart: bool = True):
    top = res["top_class"]
    conf = res["top_conf"]
    st.markdown(
        f"""
        <div class="result-card" style="background:{COLORS[top]};">
            <div class="label">Prediction</div>
            <div class="cls">{GUIDE[top]['title']}</div>
            <div class="conf">{conf*100:.1f}% confidence</div>
            <div class="tag">{GUIDE[top]['tag']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='margin-top:14px;color:#6B5F3E;font-weight:500;'>{GUIDE[top]['desc']}</div>",
        unsafe_allow_html=True,
    )
    render_uncertainty_warning(res)
    if show_chart:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-sub' style='font-weight:700;color:#1F1B12;'>Confidence across all classes</div>",
            unsafe_allow_html=True,
        )
        st.altair_chart(confidence_chart(res["probs"]), width="stretch")


def render_no_banana_panel():
    st.markdown(
        """
        <div class="nobanana">
            <div class="big">🚫 No banana detected</div>
            <div style="margin-top:8px;font-weight:600;">
                YOLOv8n found no banana with ≥40% confidence. Point the camera at a clear, well-lit banana.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_result(image: Image.Image, pipeline_result: dict):
    if not pipeline_result["banana_detected"]:
        annotated = draw_no_banana_pil(image)
        col_img, col_info = st.columns([1, 1], gap="large")
        with col_img:
            st.image(annotated, width="stretch")
        with col_info:
            render_no_banana_panel()
        return

    det = pipeline_result["detection"]
    res = pipeline_result["ripeness"]
    annotated = draw_bbox_on_pil(image, det, res, COLORS)
    col_img, col_info = st.columns([1, 1], gap="large")
    with col_img:
        st.image(annotated, width="stretch")
    with col_info:
        st.caption(f"YOLO detection: {det['conf']*100:.1f}% confidence")
        render_ripeness_panel(res)


def render_guide():
    st.markdown("<div class='section-h'>Ripeness guide</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>What each class means in practice.</div>", unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, cls in zip(cols, CLASSES):
        g = GUIDE[cls]
        with col:
            st.markdown(
                f"""
                <div class="guide-card" style="--c:{COLORS[cls]};">
                    <div class="gtag">{g['tag']}</div>
                    <div class="gt">{g['title']}</div>
                    <div class="gd">{g['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_history():
    st.markdown("<div class='section-h'>Training history</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Accuracy and loss over training epochs (training vs. validation).</div>",
        unsafe_allow_html=True,
    )
    try:
        h = get_history()
    except Exception as e:
        st.error(f"Could not load training history: {e}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Epochs trained", len(h["accuracy"]))
    m2.metric("Final train accuracy", f"{h['accuracy'][-1]*100:.2f}%")
    m3.metric("Final val accuracy", f"{h['val_accuracy'][-1]*100:.2f}%")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.altair_chart(history_chart(h["accuracy"], h["val_accuracy"], "Accuracy", "%"), width="stretch")
    with c2:
        st.altair_chart(history_chart(h["loss"], h["val_loss"], "Loss", ".2f"), width="stretch")


@st.fragment(run_every=1.0)
def render_live_feed_stats(webrtc_ctx):
    """Refresh probability chart + warnings while the live feed is running."""
    if not webrtc_ctx.state.playing:
        return
    processor = webrtc_ctx.video_processor
    if processor is None or processor.last_result is None:
        st.caption("Waiting for first detection…")
        return

    result = processor.last_result
    if not result["banana_detected"]:
        st.info("No banana in frame — point camera at banana, ensure the object is still, directly in front of camera.")
        return

    res = result["ripeness"]
    det = result["detection"]
    st.caption(f"YOLO banana detection: {det['conf']*100:.1f}% confidence")
    render_ripeness_panel(res, show_chart=True)


def render_camera_tab():
    st.markdown("<div class='section-h'>Live camera</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Real-time two-stage pipeline: "
        "YOLOv8n detects the banana, then MobileNetV2 classifies ripeness.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipeline-badge">Stage 1: YOLOv8n (COCO class 46) &nbsp;→&nbsp; '
        "Stage 2: MobileNetV2 ripeness</div>",
        unsafe_allow_html=True,
    )

    load_pipeline()  # preload models before WebRTC starts

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
        st.info("▶️ Press **Start** above to open the live camera feed.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    inject_css()
    render_header()
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    tab_cam, tab_up, tab_guide, tab_hist = st.tabs(
        ["📷  Live camera", "🖼️  Upload", "📖  Ripeness guide", "📈  Training history"]
    )

    with tab_cam:
        render_camera_tab()

    with tab_up:
        st.markdown("<div class='section-h'>Upload mode</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-sub'>Upload a photo — YOLOv8n finds the banana, then ripeness is classified.</div>",
            unsafe_allow_html=True,
        )
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"],
                              key="up", label_visibility="collapsed")
        if up is not None:
            img = Image.open(up)
            with st.spinner("Analyzing…"):
                result = load_pipeline().process_pil(img)
            render_upload_result(img, result)
        else:
            st.info("⬆️ Drag & drop or browse to upload a banana photo.")

    with tab_guide:
        render_guide()

    with tab_hist:
        render_history()

    st.markdown(
        "<div style='text-align:center;color:#A89A72;font-size:.82rem;margin-top:42px;'>"
        "YOLOv8n + MobileNetV2 • 4-class banana ripeness • streamlit-webrtc</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

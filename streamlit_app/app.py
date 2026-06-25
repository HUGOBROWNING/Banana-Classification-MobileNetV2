"""Banana Ripeness Classifier - Streamlit app.

Snapshot camera + image upload modes powered by a MobileNetV2 transfer-learning
model. Clean light theme, mobile friendly.
"""

import pickle
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from classifier import BananaRipenessClassifier, CLASSES, ASSETS_DIR
import os

# --------------------------------------------------------------------------- #
# Config & constants
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Banana Ripeness Classifier",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CONF_THRESHOLD = 0.40       # below -> "No banana detected"
UNCERTAIN_THRESHOLD = 0.75  # below -> show uncertainty warning + top 2

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


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_classifier():
    return BananaRipenessClassifier()


@st.cache_data
def get_history():
    path = os.path.join(ASSETS_DIR, "training_history.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


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

            /* Hero */
            .hero {
                background: linear-gradient(135deg, #FFF7DC 0%, #FDEFC0 55%, #FBE3A0 100%);
                border: 1px solid #F2DfA0;
                border-radius: 26px;
                padding: 38px 40px;
                box-shadow: 0 18px 40px -24px rgba(180, 140, 20, 0.55);
                position: relative;
                overflow: hidden;
            }
            .hero-title { font-size: clamp(2rem, 4.5vw, 3.2rem); font-weight: 700; color: #1F1B12; margin: 4px 0 6px; line-height: 1.05; }
            .hero-sub { font-size: 1.02rem; color: #6B5F3E; max-width: 640px; font-weight: 500; }
            .hero-badge {
                display: inline-flex; align-items: center; gap: 8px;
                background: #1F1B12; color: #FFD23F; font-weight: 700; font-size: .8rem;
                padding: 7px 14px; border-radius: 999px; letter-spacing: .04em; text-transform: uppercase;
            }

            /* Stat cards */
            .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 18px; }
            .stat {
                background: #FFFFFF; border: 1px solid #EFE7D2; border-radius: 18px; padding: 18px 20px;
                box-shadow: 0 10px 26px -22px rgba(80,60,10,.6);
            }
            .stat .v { font-family: 'Fraunces', serif; font-size: 1.7rem; font-weight: 700; color: #1F1B12; }
            .stat .l { font-size: .8rem; color: #8A7C58; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; margin-top: 2px; }

            /* Result cards */
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

            /* Guide cards */
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
# Image overlay
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def annotate_image(pil_img: Image.Image, text: str, color_hex: str) -> Image.Image:
    img = pil_img.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    fsize = max(18, int(W * 0.052))
    try:
        font = ImageFont.truetype(FONT_PATH, fsize)
    except Exception:
        font = ImageFont.load_default()

    pad = int(fsize * 0.5)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bar_h = th + pad * 2

    rgb = _hex_to_rgb(color_hex)
    draw.rectangle([0, H - bar_h, W, H], fill=(rgb[0], rgb[1], rgb[2], 235))
    draw.text(((W - tw) / 2, H - bar_h + pad - bbox[1]), text, font=font, fill=(255, 255, 255, 255))
    return img


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
    line = alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1, title="Epoch")),
        y=alt.Y("Value:Q", axis=alt.Axis(format=fmt, title=None)),
        color=alt.Color("Set:N", scale=alt.Scale(domain=["Training", "Validation"], range=["#E29400", "#4C9A2A"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Epoch", alt.Tooltip("Value:Q", format=".3f"), "Set"],
    ).properties(height=280, title=title).configure_view(strokeOpacity=0)
    return line


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_header():
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">● 98.22% test accuracy</span>
            <div class="hero-title">Banana Ripeness Classifier</div>
            <div class="hero-sub">Snap or upload a photo of a banana and an AI model trained on thousands of images
            tells you whether it's unripe, ripe, overripe or rotten — with a live confidence breakdown.</div>
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


def render_result(image: Image.Image, res: dict):
    top = res["top_class"]
    conf = res["top_conf"]

    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        if res["no_banana"]:
            annotated = annotate_image(image, "No banana detected", "#9A3B3B")
        else:
            annotated = annotate_image(image, f"{GUIDE[top]['title']}  {conf*100:.1f}%", COLORS[top])
        st.image(annotated, use_container_width=True)

    with col_info:
        if res["no_banana"]:
            reason = ("Top confidence below 40%" if conf < CONF_THRESHOLD
                      else "Image doesn't look like a banana")
            st.markdown(
                f"""
                <div class="nobanana">
                    <div class="big">🚫 No banana detected</div>
                    <div style="margin-top:8px;font-weight:600;">{reason}. Try a clearer, well-lit photo of a single banana.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
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

    # Uncertainty handling
    if res["uncertain"]:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.warning(
            f"⚠️ **Uncertain result** — top confidence is {conf*100:.1f}% (below 75%). "
            "Overripe and rotten bananas look similar, so here are the two most likely classes:"
        )
        (c1, c2) = res["ordered"][0], res["ordered"][1]
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

    # Confidence bar chart (always shown)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub' style='font-weight:700;color:#1F1B12;'>Confidence across all classes</div>",
                unsafe_allow_html=True)
    st.altair_chart(confidence_chart(res["probs"]), use_container_width=True)


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
    st.markdown("<div class='section-sub'>Accuracy and loss over training epochs (training vs. validation).</div>",
                unsafe_allow_html=True)
    try:
        h = get_history()
    except Exception as e:
        st.error(f"Could not load training history: {e}")
        return

    final_acc = h["accuracy"][-1] * 100
    final_val = h["val_accuracy"][-1] * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("Epochs trained", len(h["accuracy"]))
    m2.metric("Final train accuracy", f"{final_acc:.2f}%")
    m3.metric("Final val accuracy", f"{final_val:.2f}%")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.altair_chart(history_chart(h["accuracy"], h["val_accuracy"], "Accuracy", "%"),
                        use_container_width=True)
    with c2:
        st.altair_chart(history_chart(h["loss"], h["val_loss"], "Loss", ".2f"),
                        use_container_width=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    inject_css()
    render_header()
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    clf = get_classifier()

    tab_cam, tab_up, tab_guide, tab_hist = st.tabs(
        ["📷  Camera", "🖼️  Upload", "📖  Ripeness guide", "📈  Training history"]
    )

    with tab_cam:
        st.markdown("<div class='section-h'>Camera mode</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Take a snapshot of a banana — the result is overlaid on the photo.</div>",
                    unsafe_allow_html=True)
        shot = st.camera_input("Point at a banana and capture", key="cam", label_visibility="collapsed")
        if shot is not None:
            img = Image.open(shot)
            with st.spinner("Analyzing…"):
                res = clf.predict(img, CONF_THRESHOLD, UNCERTAIN_THRESHOLD)
            render_result(img, res)
        else:
            st.info("📸 Allow camera access and capture a photo to classify.")

    with tab_up:
        st.markdown("<div class='section-h'>Upload mode</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Upload a JPG or PNG image of a banana.</div>",
                    unsafe_allow_html=True)
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"],
                              key="up", label_visibility="collapsed")
        if up is not None:
            img = Image.open(up)
            with st.spinner("Analyzing…"):
                res = clf.predict(img, CONF_THRESHOLD, UNCERTAIN_THRESHOLD)
            render_result(img, res)
        else:
            st.info("⬆️ Drag & drop or browse to upload a banana photo.")

    with tab_guide:
        render_guide()

    with tab_hist:
        render_history()

    st.markdown(
        "<div style='text-align:center;color:#A89A72;font-size:.82rem;margin-top:42px;'>"
        "Built with Streamlit • MobileNetV2 transfer learning • 4-class banana ripeness model</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

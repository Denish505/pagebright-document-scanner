"""Local Streamlit application for document image enhancement."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from docscan.pipeline import process_document  # noqa: E402


st.set_page_config(
    page_title="PageBright | Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1420px;}
      .hero {padding: 1.5rem 1.7rem; border: 1px solid #dbe5ee; border-radius: 18px;
             background: linear-gradient(120deg, #f4f9ff 0%, #f8fbfa 100%); margin-bottom: 1.2rem;}
      .hero h1 {font-size: 2.2rem; margin: 0; color: #15314b;}
      .hero p {font-size: 1.03rem; color: #526678; margin: .55rem 0 0 0;}
      div[data-testid="stMetric"] {background: #f5f8fb; border: 1px solid #e1e9ef;
                                    padding: .8rem 1rem; border-radius: 12px;}
      .small-note {color: #64788a; font-size: .9rem;}
    </style>
    <div class="hero">
      <h1>PageBright</h1>
      <p>Turn a phone photo into a clean, straightened document with PageBright.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _pil_rgb(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _make_pdf(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PDF", resolution=150.0)
    return output.getvalue()


def _make_bundle(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()


with st.sidebar:
    st.header("Scan settings")
    mode_label = st.selectbox(
        "Output styles",
        ("All styles", "Color only", "Grayscale only", "Black and white only"),
        help="All styles creates color, grayscale, and black-and-white versions.",
    )
    mode = {
        "All styles": "all",
        "Color only": "color",
        "Grayscale only": "gray",
        "Black and white only": "bw",
    }[mode_label]
    denoise = st.toggle("Reduce image noise", value=True)
    min_area = st.slider(
        "Smallest page size in photo",
        min_value=5,
        max_value=40,
        value=8,
        step=1,
        format="%d%%",
        help="Raise this if the detector selects a smaller object instead of the page.",
    ) / 100.0
    st.divider()
    with st.expander("Manual page corners", expanded=False):
        st.caption("Use this if automatic page detection fails. Coordinates start at the top-left of the uploaded image.")
        manual_enabled = st.checkbox("Enter four corners manually", value=False)
        manual_corners = None
        if manual_enabled:
            dims = st.session_state.get("pagebright_dimensions", (1000, 1400))
            iw, ih = dims
            initial = [
                (int(iw * 0.05), int(ih * 0.05)),
                (int(iw * 0.95), int(ih * 0.05)),
                (int(iw * 0.95), int(ih * 0.95)),
                (int(iw * 0.05), int(ih * 0.95)),
            ]
            labels = ("Top left", "Top right", "Bottom right", "Bottom left")
            points = []
            for index, label in enumerate(labels):
                c1, c2 = st.columns(2)
                with c1:
                    x = st.number_input(f"{label} X", min_value=0, max_value=max(1, iw - 1), value=initial[index][0], key=f"corner_x_{index}")
                with c2:
                    y = st.number_input(f"{label} Y", min_value=0, max_value=max(1, ih - 1), value=initial[index][1], key=f"corner_y_{index}")
                points.append((x, y))
            manual_corners = np.asarray(points, dtype=np.float32)

    st.caption("Tip: use a well-lit photo with all four page edges visible for best results.")


uploaded = st.file_uploader(
    "Choose a document photo",
    type=("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"),
    help="Supported image formats: JPG, PNG, WEBP, BMP, and TIFF.",
)

if uploaded is None:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("A cleaner scan in three steps")
        st.markdown(
            "1. Upload a photo of one paper page.\n"
            "2. Let PageBright find the page edges and flatten perspective.\n"
            "3. Download the enhanced images or a ready-to-share PDF."
        )
        st.info("Privacy note: uploaded images are sent to this hosted app for processing. Temporary working files are removed when processing completes. Avoid uploading sensitive documents.")
    with right:
        st.subheader("Included tools")
        st.markdown("- Automatic page-edge detection\n- Manual corner fallback\n- Perspective correction\n- Color, grayscale, and high-contrast output\n- One-click PDF and ZIP downloads")
else:
    uploaded_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    decoded = cv2.imdecode(np.frombuffer(uploaded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        st.error("This file could not be opened as an image. Try saving it as JPG or PNG and uploading again.")
        st.stop()

    image_h, image_w = decoded.shape[:2]
    st.session_state["pagebright_dimensions"] = (image_w, image_h)
    st.caption(f"{uploaded.name}  ·  {image_w:,} × {image_h:,} pixels")

    preview_col, action_col = st.columns([1.25, 1])
    with preview_col:
        st.image(_pil_rgb(decoded), caption="Original photo", use_container_width=True)
    with action_col:
        st.subheader("Ready to scan")
        st.markdown("PageBright will locate the page boundary, correct the camera angle, and prepare your selected output styles.")
        process_clicked = st.button("Enhance document", type="primary", use_container_width=True)

    current_settings = {
        "file": file_hash,
        "mode": mode,
        "denoise": denoise,
        "min_area": min_area,
        "manual": manual_corners.tolist() if manual_corners is not None else None,
    }
    settings_hash = hashlib.sha256(json.dumps(current_settings, sort_keys=True).encode("utf-8")).hexdigest()
    if st.session_state.get("pagebright_settings_hash") != settings_hash:
        st.session_state.pop("pagebright_result", None)

    if process_clicked:
        try:
            with st.spinner("Finding page edges and preparing your scan…"):
                with tempfile.TemporaryDirectory(prefix="pagebright-") as temp_dir:
                    temp_root = Path(temp_dir)
                    safe_name = Path(uploaded.name).name
                    input_path = temp_root / safe_name
                    input_path.write_bytes(uploaded_bytes)
                    output_dir = temp_root / "outputs"
                    report = process_document(
                        input_path,
                        output_dir,
                        mode=mode,
                        manual_corners=manual_corners,
                        min_area_fraction=min_area,
                        denoise=denoise,
                    )
                    files = {name: (output_dir / name).read_bytes() for name in report["outputs"]}
                    pdf_source = "enhanced_color.jpg" if "enhanced_color.jpg" in files else "rectified_color.jpg"
                    files["document_scan.pdf"] = _make_pdf(files[pdf_source])
                    files["metadata.json"] = (output_dir / "metadata.json").read_bytes()
                    bundle = _make_bundle(files)
            st.session_state["pagebright_result"] = {"report": report, "files": files, "bundle": bundle}
            st.session_state["pagebright_settings_hash"] = settings_hash
        except (FileNotFoundError, OSError, ValueError, cv2.error) as error:
            st.error(str(error))
            if manual_corners is None:
                st.info("If the page edges are hard to distinguish, open **Manual page corners** in the sidebar and enter the four page corners.")

    result = st.session_state.get("pagebright_result")
    if result and st.session_state.get("pagebright_settings_hash") == settings_hash:
        report = result["report"]
        files = result["files"]
        st.divider()
        st.subheader("Your scan is ready")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Page detection", report["detection_method"].replace("_", " ").title())
        metric2.metric("Straightened image", f"{report['rectified_size_px']['width']:,} × {report['rectified_size_px']['height']:,} px")
        metric3.metric("Processing time", f"{report['processing_seconds']:.2f} sec")

        tabs = st.tabs(["Boundary preview", "Color", "Grayscale", "Black and white", "Details"])
        tab_files = (
            ("boundary_preview.jpg", "Detected page boundary"),
            ("enhanced_color.jpg", "Enhanced color scan"),
            ("enhanced_gray.jpg", "Enhanced grayscale scan"),
            ("enhanced_bw.png", "High-contrast scan"),
        )
        for tab, (filename, caption) in zip(tabs[:4], tab_files):
            with tab:
                if filename in files:
                    image = cv2.imdecode(np.frombuffer(files[filename], dtype=np.uint8), cv2.IMREAD_COLOR)
                    st.image(_pil_rgb(image), caption=caption, use_container_width=True)
                    st.download_button(
                        f"Download {filename}", files[filename], file_name=filename,
                        mime="image/png" if filename.endswith(".png") else "image/jpeg",
                        key=f"download_{filename}",
                    )
                else:
                    st.info("This style was not selected. Choose **All styles** or change the output setting in the sidebar and run again.")
        with tabs[4]:
            st.json(report)

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download scan as PDF",
                files["document_scan.pdf"],
                file_name=f"{Path(uploaded.name).stem}_scan.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                "Download all results as ZIP",
                result["bundle"],
                file_name=f"{Path(uploaded.name).stem}_pagebright_results.zip",
                mime="application/zip",
                use_container_width=True,
            )
        st.caption("The PDF contains the color-enhanced page when available; otherwise it contains the rectified page.")


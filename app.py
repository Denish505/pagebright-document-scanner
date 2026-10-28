"""Streamlit app for single-image and multi-page PDF document enhancement."""
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

from docscan.geometry import order_corners  # noqa: E402
from docscan.pdf import get_pdf_page_count, render_pdf_page  # noqa: E402
from docscan.pipeline import process_document  # noqa: E402

MODEL_PATH = ROOT / "models" / "pagebright_corners.onnx"

st.set_page_config(
    page_title="PageBright | Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1420px;}
      .hero {padding: 1.4rem 1.7rem; border: 1px solid #dbe5ee; border-radius: 18px;
             background: linear-gradient(120deg, #f4f9ff 0%, #f8fbfa 100%); margin-bottom: 1.2rem;}
      .hero h1 {font-size: 2.15rem; margin: 0; color: #15314b;}
      .hero p {font-size: 1rem; color: #526678; margin: .55rem 0 0 0;}
      div[data-testid="stMetric"] {background: #f5f8fb; border: 1px solid #e1e9ef;
                                    padding: .8rem 1rem; border-radius: 12px;}
    </style>
    <div class="hero">
      <h1>PageBright</h1>
      <p>Clean up document photos and scanned PDFs, one page or an entire packet at a time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "Hosted privacy: uploads are processed on the hosting server and are not added to a shared training set. "
    "Processing uses temporary storage. Avoid sensitive documents."
)

uploaded = st.file_uploader(
    "Choose a document photo or PDF",
    type=("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "pdf"),
    help="PDFs are processed page by page, up to 20 pages per file.",
)

is_pdf = bool(uploaded and uploaded.name.lower().endswith(".pdf"))
uploaded_bytes = uploaded.getvalue() if uploaded else b""
file_hash = hashlib.sha256(uploaded_bytes).hexdigest() if uploaded else ""

first_page = None
page_count = 1
pdf_dpi = 144
detect_pdf_edges = False
if uploaded:
    try:
        if is_pdf:
            pdf_dpi = st.sidebar.slider(
                "PDF render quality (DPI)", 96, 200, 144, 12,
                help="Higher resolution preserves small text but uses more memory.",
            )
            page_count = get_pdf_page_count(uploaded_bytes)
            first_page = render_pdf_page(uploaded_bytes, 0, pdf_dpi)
            detect_pdf_edges = st.checkbox(
                "Detect and crop a photographed page inside the PDF",
                value=False,
                help="Leave off for normal scanned PDFs. Turn on when each page contains a photo of paper.",
            )
        else:
            first_page = cv2.imdecode(
                np.frombuffer(uploaded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if first_page is None:
                raise ValueError("This file could not be opened as an image. Try JPG or PNG.")
    except (ValueError, OSError, cv2.error) as error:
        st.error(str(error))
        st.stop()
    h, w = first_page.shape[:2]
    if st.session_state.get("corner_image_hash") != file_hash:
        for index in range(4):
            st.session_state.pop(f"corner_x_{index}", None)
            st.session_state.pop(f"corner_y_{index}", None)
        st.session_state["corner_image_hash"] = file_hash
    st.session_state["pagebright_dimensions"] = (w, h)

with st.sidebar:
    st.header("Scan settings")
    mode_label = st.selectbox(
        "Output styles",
        ("All styles", "Color only", "Grayscale only", "Black and white only"),
        help="All styles creates color, grayscale, and high-contrast images.",
    )
    mode = {
        "All styles": "all",
        "Color only": "color",
        "Grayscale only": "gray",
        "Black and white only": "bw",
    }[mode_label]
    denoise = st.toggle("Reduce image noise", value=True)
    min_area = st.slider(
        "Smallest page size in photo", 5, 40, 8, 1, format="%d%%",
        help="Raise this if detection selects a smaller object.",
    ) / 100.0
    with st.expander("Manual page corners", expanded=False):
        st.caption("Coordinates refer to the page preview. Use this if automatic detection misses an edge.")
        manual_enabled = st.checkbox("Use my four corner points", value=False)
        manual_corners = None
        if manual_enabled:
            iw, ih = st.session_state.get("pagebright_dimensions", (1000, 1400))
            initial = [
                (int(iw * .04), int(ih * .04)),
                (int(iw * .96), int(ih * .04)),
                (int(iw * .96), int(ih * .96)),
                (int(iw * .04), int(ih * .96)),
            ]
            labels = ("Top left", "Top right", "Bottom right", "Bottom left")
            points = []
            for index, label in enumerate(labels):
                c1, c2 = st.columns(2)
                with c1:
                    x = st.number_input(
                        f"{label} X", 0, max(1, iw - 1),
                        min(initial[index][0], iw - 1), key=f"corner_x_{index}",
                    )
                with c2:
                    y = st.number_input(
                        f"{label} Y", 0, max(1, ih - 1),
                        min(initial[index][1], ih - 1), key=f"corner_y_{index}",
                    )
                points.append((x, y))
            manual_corners = np.asarray(points, dtype=np.float32)
    st.caption(
        "A trained corner model is used if models/pagebright_corners.onnx exists. "
        "Otherwise, PageBright uses multiple edge and contour methods."
    )

if uploaded is None:
    st.subheader("What PageBright can do")
    st.markdown(
        "- Scan photos and multi-page PDFs\n"
        "- Correct perspective and uneven lighting\n"
        "- Export color, grayscale, or high-contrast pages\n"
        "- Download a combined PDF or ZIP of every page\n"
        "- Export corrected examples to help train a corner model"
    )
    st.info("Choose a file above to get started.")
    st.stop()

h, w = first_page.shape[:2]
st.caption(
    f"{uploaded.name} · {page_count} page(s) · {w:,} × {h:,} px"
    + (f" · {pdf_dpi} DPI" if is_pdf else "")
)
preview_col, action_col = st.columns([1.2, 1])
with preview_col:
    st.image(
        cv2.cvtColor(first_page, cv2.COLOR_BGR2RGB),
        caption="First PDF page" if is_pdf else "Original photo",
        use_container_width=True,
    )
with action_col:
    st.subheader("Ready to process")
    if is_pdf:
        st.markdown(f"Enhance all **{page_count} pages**, then download one combined PDF or all page outputs.")
    else:
        st.markdown("Find the page, flatten its perspective, and prepare the selected output styles.")
    process_clicked = st.button(
        f"Enhance {page_count} page(s)" if is_pdf else "Enhance document",
        type="primary",
        use_container_width=True,
    )

settings = {
    "file": file_hash,
    "mode": mode,
    "denoise": denoise,
    "min_area": min_area,
    "dpi": pdf_dpi if is_pdf else None,
    "detect_pdf_edges": detect_pdf_edges if is_pdf else None,
    "manual": manual_corners.tolist() if manual_corners is not None else None,
}
settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
if st.session_state.get("pagebright_settings_hash") != settings_hash:
    st.session_state.pop("pagebright_result", None)

if process_clicked:
    output_files: dict[str, bytes] = {}
    page_reports: list[dict] = []
    pdf_page_images: list[Image.Image] = []
    notices: list[str] = []
    try:
        with st.spinner(f"Processing {page_count} page(s)…"):
            with tempfile.TemporaryDirectory(prefix="pagebright-") as temp_dir:
                temp_root = Path(temp_dir)
                for page_index in range(page_count):
                    image = (
                        render_pdf_page(uploaded_bytes, page_index, pdf_dpi)
                        if is_pdf
                        else first_page
                    )
                    page_input = temp_root / f"page_{page_index + 1:03d}.png"
                    if not cv2.imwrite(str(page_input), image):
                        raise OSError("Could not prepare a page for processing.")

                    corners_for_page = None
                    if is_pdf and page_index > 0 and not detect_pdf_edges:
                        ih, iw = image.shape[:2]
                        corners_for_page = np.array(
                            [[0, 0], [iw - 1, 0], [iw - 1, ih - 1], [0, ih - 1]],
                            dtype=np.float32,
                        )
                    elif is_pdf and page_index == 0 and not detect_pdf_edges and manual_corners is None:
                        ih, iw = image.shape[:2]
                        corners_for_page = np.array(
                            [[0, 0], [iw - 1, 0], [iw - 1, ih - 1], [0, ih - 1]],
                            dtype=np.float32,
                        )
                    elif manual_corners is not None and page_index == 0:
                        corners_for_page = manual_corners

                    page_dir = temp_root / f"output_{page_index + 1:03d}"
                    try:
                        report = process_document(
                            page_input,
                            page_dir,
                            mode=mode,
                            manual_corners=corners_for_page,
                            min_area_fraction=min_area,
                            denoise=denoise,
                            model_path=MODEL_PATH if MODEL_PATH.exists() else None,
                        )
                    except ValueError:
                        if not is_pdf or not detect_pdf_edges:
                            raise
                        ih, iw = image.shape[:2]
                        full_page = np.array(
                            [[0, 0], [iw - 1, 0], [iw - 1, ih - 1], [0, ih - 1]],
                            dtype=np.float32,
                        )
                        report = process_document(
                            page_input, page_dir, mode=mode, manual_corners=full_page,
                            min_area_fraction=min_area, denoise=denoise,
                        )
                        report["detection_method"] = "full_page_fallback"
                        notices.append(
                            f"Page {page_index + 1}: no internal page edge was found, so the full page was kept."
                        )
                    if is_pdf and not detect_pdf_edges and corners_for_page is not None:
                        report["detection_method"] = "pdf_full_page"
                    report["page_number"] = page_index + 1
                    page_reports.append(report)
                    prefix = f"page_{page_index + 1:03d}/"
                    for output_name in report["outputs"]:
                        output_files[prefix + output_name] = (page_dir / output_name).read_bytes()

                    if "enhanced_color.jpg" in report["outputs"]:
                        page_pdf = Image.open(page_dir / "enhanced_color.jpg").convert("RGB")
                    elif "enhanced_gray.jpg" in report["outputs"]:
                        page_pdf = Image.open(page_dir / "enhanced_gray.jpg").convert("RGB")
                    elif "enhanced_bw.png" in report["outputs"]:
                        page_pdf = Image.open(page_dir / "enhanced_bw.png").convert("RGB")
                    else:
                        page_pdf = Image.open(page_dir / "rectified_color.jpg").convert("RGB")
                    pdf_page_images.append(page_pdf.copy())
                    page_pdf.close()

        pdf_buffer = io.BytesIO()
        pdf_page_images[0].save(
            pdf_buffer,
            format="PDF",
            save_all=True,
            append_images=pdf_page_images[1:],
            resolution=float(pdf_dpi if is_pdf else 150),
        )
        output_files["document_scan.pdf"] = pdf_buffer.getvalue()
        metadata = {
            "source_file": Path(uploaded.name).name,
            "source_type": "pdf" if is_pdf else "image",
            "page_count": page_count,
            "pdf_render_dpi": pdf_dpi if is_pdf else None,
            "pages": page_reports,
            "notices": notices,
        }
        output_files["metadata.json"] = json.dumps(metadata, indent=2).encode("utf-8")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in output_files.items():
                archive.writestr(name, data)
        st.session_state["pagebright_result"] = {
            "reports": page_reports,
            "files": output_files,
            "bundle": zip_buffer.getvalue(),
            "notices": notices,
            "settings_hash": settings_hash,
        }
    except (FileNotFoundError, OSError, ValueError, cv2.error, RuntimeError) as error:
        st.error(str(error))
        if manual_corners is None and not (is_pdf and not detect_pdf_edges):
            st.info("If automatic detection misses a page edge, open **Manual page corners** and enter the four points.")

result = st.session_state.get("pagebright_result")
if result and result.get("settings_hash") == settings_hash:
    reports, files = result["reports"], result["files"]
    for notice in result["notices"]:
        st.warning(notice)
    st.divider()
    st.subheader("Your scan is ready")
    metric1, metric2, metric3 = st.columns(3)
    methods = sorted({r["detection_method"].replace("_", " ").title() for r in reports})
    metric1.metric("Pages processed", len(reports))
    metric2.metric("Detection", ", ".join(methods))
    metric3.metric("Processing time", f"{sum(r['processing_seconds'] for r in reports):.2f} sec")

    selected_page = st.selectbox(
        "Preview page", list(range(1, len(reports) + 1)),
        format_func=lambda value: f"Page {value}",
    )
    prefix = f"page_{selected_page:03d}/"
    with st.expander("Correct this page and update the PDF", expanded=False):
        st.caption(
            "You can fix a page after the full PDF has been enhanced. Set its four page corners, "
            "then update just this page in the combined PDF."
        )
        original_page = (
            render_pdf_page(uploaded_bytes, selected_page - 1, pdf_dpi)
            if is_pdf else first_page
        )
        page_h, page_w = original_page.shape[:2]
        st.image(
            cv2.cvtColor(original_page, cv2.COLOR_BGR2RGB),
            caption=f"Original page {selected_page} — use these coordinates",
            use_container_width=True,
        )
        repair_points = []
        repair_labels = ("Top left", "Top right", "Bottom right", "Bottom left")
        repair_defaults = (
            (int(page_w * .04), int(page_h * .04)),
            (int(page_w * .96), int(page_h * .04)),
            (int(page_w * .96), int(page_h * .96)),
            (int(page_w * .04), int(page_h * .96)),
        )
        repair_key = f"{file_hash[:12]}_{selected_page}"
        for point_index, (label, default) in enumerate(zip(repair_labels, repair_defaults)):
            x_col, y_col = st.columns(2)
            with x_col:
                repair_x = st.number_input(
                    f"{label} X", min_value=0, max_value=max(1, page_w - 1),
                    value=min(default[0], page_w - 1),
                    key=f"repair_x_{repair_key}_{point_index}",
                )
            with y_col:
                repair_y = st.number_input(
                    f"{label} Y", min_value=0, max_value=max(1, page_h - 1),
                    value=min(default[1], page_h - 1),
                    key=f"repair_y_{repair_key}_{point_index}",
                )
            repair_points.append((repair_x, repair_y))
        if st.button("Update this page in the combined PDF", key=f"repair_page_{repair_key}"):
            try:
                with tempfile.TemporaryDirectory(prefix="pagebright-repair-") as repair_dir:
                    repair_root = Path(repair_dir)
                    source_path = repair_root / "source.png"
                    page_output_dir = repair_root / "output"
                    if not cv2.imwrite(str(source_path), original_page):
                        raise OSError("Could not prepare the selected page for correction.")
                    corrected_report = process_document(
                        source_path, page_output_dir, mode=mode,
                        manual_corners=np.asarray(repair_points, dtype=np.float32),
                        min_area_fraction=min_area, denoise=denoise,
                        model_path=MODEL_PATH if MODEL_PATH.exists() else None,
                    )
                    corrected_report["page_number"] = selected_page
                    for old_name in list(files):
                        if old_name.startswith(prefix):
                            del files[old_name]
                    for output_name in corrected_report["outputs"]:
                        files[prefix + output_name] = (page_output_dir / output_name).read_bytes()
                    reports[selected_page - 1] = corrected_report
                    pdf_images = []
                    for page_number in range(1, len(reports) + 1):
                        page_prefix = f"page_{page_number:03d}/"
                        image_name = next(
                            (page_prefix + name for name in (
                                "enhanced_color.jpg", "enhanced_gray.jpg",
                                "enhanced_bw.png", "rectified_color.jpg",
                            ) if page_prefix + name in files),
                            None,
                        )
                        if image_name is None:
                            raise ValueError(f"No processed image was saved for page {page_number}.")
                        with Image.open(io.BytesIO(files[image_name])) as page_image:
                            pdf_images.append(page_image.convert("RGB"))
                    pdf_buffer = io.BytesIO()
                    pdf_images[0].save(
                        pdf_buffer, format="PDF", save_all=True,
                        append_images=pdf_images[1:],
                        resolution=float(pdf_dpi if is_pdf else 150),
                    )
                    files["document_scan.pdf"] = pdf_buffer.getvalue()
                    for page_image in pdf_images:
                        page_image.close()
                    metadata = {
                        "source_file": Path(uploaded.name).name,
                        "source_type": "pdf" if is_pdf else "image",
                        "page_count": len(reports),
                        "pdf_render_dpi": pdf_dpi if is_pdf else None,
                        "pages": reports,
                        "notices": result["notices"],
                    }
                    files["metadata.json"] = json.dumps(metadata, indent=2).encode("utf-8")
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        for name, data in files.items():
                            archive.writestr(name, data)
                    result["bundle"] = zip_buffer.getvalue()
                    st.session_state["pagebright_result"] = result
                    st.success(f"Page {selected_page} was updated in the combined PDF and ZIP.")
            except (OSError, ValueError, cv2.error, RuntimeError) as error:
                st.error(f"Could not update page {selected_page}: {error}")
    tabs = st.tabs(["Boundary", "Color", "Grayscale", "Black and white", "Details"])
    for tab, filename, caption in (
        (tabs[0], "boundary_preview.jpg", "Detected page boundary"),
        (tabs[1], "enhanced_color.jpg", "Enhanced color scan"),
        (tabs[2], "enhanced_gray.jpg", "Enhanced grayscale scan"),
        (tabs[3], "enhanced_bw.png", "High-contrast scan"),
    ):
        with tab:
            name = prefix + filename
            if name in files:
                decoded = cv2.imdecode(np.frombuffer(files[name], dtype=np.uint8), cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)
                st.download_button(
                    f"Download page {selected_page} {filename}", files[name],
                    file_name=f"page_{selected_page:03d}_{filename}",
                    mime="image/png" if filename.endswith(".png") else "image/jpeg",
                    key=f"download_{selected_page}_{filename}",
                )
            else:
                st.info("This output style was not selected.")
    with tabs[4]:
        st.json(reports[selected_page - 1])

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download combined PDF", files["document_scan.pdf"],
            file_name=f"{Path(uploaded.name).stem}_pagebright.pdf",
            mime="application/pdf", type="primary", use_container_width=True,
        )
    with right:
        st.download_button(
            "Download all pages and details (ZIP)", result["bundle"],
            file_name=f"{Path(uploaded.name).stem}_pagebright_results.zip",
            mime="application/zip", use_container_width=True,
        )

    if manual_corners is not None:
        st.markdown("#### Export a corrected training example")
        st.caption(
            "Check the corner labels before export. This creates a ZIP on your device; "
            "PageBright does not save it or add it to a shared training set."
        )
        if st.button("Prepare labeled training sample"):
            try:
                ordered = order_corners(manual_corners)
                ok, encoded = cv2.imencode(".jpg", first_page, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise OSError("Could not encode the labeled page.")
                label = {
                    "image": "page.jpg",
                    "width": int(w),
                    "height": int(h),
                    "corners_normalized": np.round(
                        ordered / np.array([w, h], dtype=np.float32), 6
                    ).tolist(),
                }
                sample_buffer = io.BytesIO()
                with zipfile.ZipFile(sample_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("page.jpg", encoded.tobytes())
                    archive.writestr("labels.json", json.dumps(label, indent=2))
                st.download_button(
                    "Download labeled sample", sample_buffer.getvalue(),
                    file_name="pagebright_training_sample.zip",
                    mime="application/zip", key="download_training_sample",
                )
            except (ValueError, OSError) as error:
                st.error(str(error))

"""PageBright: Streamlit app for document photo and multi-page PDF enhancement."""
from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from docscan.geometry import order_corners  # noqa: E402
from docscan.pdf import get_pdf_page_count, render_pdf_page  # noqa: E402
from docscan.pipeline import (  # noqa: E402
    OUTPUT_FILENAMES, encode_image, full_page_corners, process_image,
)

MODEL_PATH = ROOT / "models" / "pagebright_corners.onnx"
DEMO_PATH = ROOT / "examples" / "demo_input.png"
STYLE_LABELS = {"color": "Color", "gray": "Grayscale", "bw": "Black and white"}
AUTO_METHODS = {"trained_model", "canny", "otsu", "adaptive"}
METHOD_LABELS = {
    "manual": "Adjusted by you",
    "pdf_full_page": "Full PDF page",
    "fallback": "Edges not found – full photo kept",
    "full_page_fallback": "Edges not found – full page kept",
    **{method: "Found automatically" for method in AUTO_METHODS},
}

# --------------------------------------------------------------------------- compatibility
# Works on Streamlit 1.37+ : newer keyword arguments are dropped or translated
# when the installed version doesn't know them.
ST_VERSION = tuple(int(part) for part in st.__version__.split(".")[:2] if part.isdigit())
STRING_WIDTH = ST_VERSION >= (1, 47)
_PARAM_CACHE: dict[str, set[str]] = {}


def _params(fn) -> set[str]:
    name = getattr(fn, "__qualname__", repr(fn))
    if name not in _PARAM_CACHE:
        try:
            _PARAM_CACHE[name] = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            _PARAM_CACHE[name] = set()
    return _PARAM_CACHE[name]


def ui(fn, *args, stretch: bool = False, **kwargs):
    """Call a Streamlit element, passing only arguments this version supports."""
    params = _params(fn)
    if stretch:
        if STRING_WIDTH and "width" in params:
            kwargs["width"] = "stretch"
        elif "use_container_width" in params:
            kwargs["use_container_width"] = True
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


_corner_editor = components.declare_component("corner_editor", path=str(ROOT / "components" / "corner_editor"))

st.set_page_config(page_title="PageBright | Document Scanner", page_icon="📄", layout="wide",
                   initial_sidebar_state="auto")

# --------------------------------------------------------------------------- styles
st.markdown(
    """
    <style>
      :root { --pb-ink:#15314b; --pb-muted:#5b6f82; --pb-line:#dde6ee; --pb-soft:#f4f8fb; --pb-teal:#147d8a; }
      .block-container {padding-top: 4rem; padding-bottom: 2rem; max-width: 1360px;}
      .pb-top {display:flex; align-items:center; gap:.9rem; flex-wrap:wrap; margin-bottom:.4rem;}
      .pb-logo {width:44px; height:44px; border-radius:12px; background:var(--pb-teal);
                display:flex; align-items:center; justify-content:center; flex:none;}
      .pb-title {font-size:1.75rem; font-weight:700; color:var(--pb-ink); line-height:1.15;}
      .pb-sub {color:var(--pb-muted); font-size:.98rem; margin-top:.15rem;}
      .pb-steps {display:flex; gap:.5rem; flex-wrap:wrap; margin:.9rem 0 1.1rem 0;}
      .pb-step {display:flex; align-items:center; gap:.5rem; padding:.42rem .85rem .42rem .45rem;
                border:1px solid var(--pb-line); border-radius:999px; color:var(--pb-muted);
                font-size:.9rem; background:#fff; white-space:nowrap; line-height:1.2;}
      .pb-step b {width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center;
                  justify-content:center; background:var(--pb-soft); color:var(--pb-muted); font-size:.8rem;}
      .pb-step.on {border-color:var(--pb-teal); color:var(--pb-ink); font-weight:600;}
      .pb-step.on b {background:var(--pb-teal); color:#fff;}
      .pb-step.done b {background:#d8efe9; color:#0f6b5c;}
      .pb-cards {display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:.8rem; margin:.6rem 0 1rem;}
      .pb-card {border:1px solid var(--pb-line); border-radius:14px; padding:1rem 1.1rem; background:#fff;}
      .pb-card h4 {margin:.45rem 0 .25rem; font-size:1rem; color:var(--pb-ink); padding:0;}
      .pb-card p {margin:0; color:var(--pb-muted); font-size:.9rem; line-height:1.45;}
      .pb-ico {width:32px; height:32px; border-radius:9px; background:#e3f2f3; display:flex;
               align-items:center; justify-content:center;}
      .pb-privacy {color:var(--pb-muted); font-size:.85rem; margin:.1rem 0 .6rem;}
      .pb-label {font-weight:600; color:var(--pb-ink); margin:.2rem 0 .35rem; font-size:1rem;}
      .pb-footer {margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--pb-line);
                  color:var(--pb-muted); font-size:.82rem; text-align:center;}
      div[data-testid="stMetric"] {background:var(--pb-soft); border:1px solid var(--pb-line);
                                    padding:.65rem 1rem; border-radius:12px;}
      div[data-testid="stMetricValue"] {font-size:1.45rem;}
      div[data-testid="stFileUploader"] section {border:1.5px dashed #b9cad8; border-radius:14px;}
      .stButton button, .stDownloadButton button {border-radius:10px; font-weight:600;}
      @media (max-width: 640px) {
        .block-container {padding-top: 3.5rem;}
        .pb-title {font-size:1.45rem;} .pb-step {font-size:.82rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

ICON_DOC = ("<svg width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' "
            "stroke-linecap='round' stroke-linejoin='round'><path d='M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 "
            "2-2V8z'/><path d='M14 3v5h5'/><path d='M9 13h6M9 17h4'/></svg>")


def _icon(path: str) -> str:
    return ("<div class='pb-ico'><svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#147d8a' "
            f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{path}</svg></div>")


FEATURES = (
    ('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>',
     "Finds the page", "Detects the sheet's four corners, even on a busy desk."),
    ('<path d="M4 20 8 4h8l4 16z"/><path d="M4 20h16"/>',
     "Flattens perspective", "Straightens tilted photos into a rectangular page."),
    ('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19'
     'M5 19l1.5-1.5M17.5 6.5 19 5"/>',
     "Removes shadows", "Evens out lighting for a white page and crisp text."),
    ('<path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 21h14"/>',
     "Exports anything", "Color, grayscale or black and white; one PDF or a ZIP."),
)


def header(step: int) -> None:
    labels = ("Upload", "Scan", "Adjust & download")
    items = "".join(
        f"<div class='pb-step {'on' if i == step else 'done' if i < step else ''}'>"
        f"<b>{'✓' if i < step else i + 1}</b><span>{label}</span></div>"
        for i, label in enumerate(labels)
    )
    st.markdown(
        f"<div class='pb-top'><div class='pb-logo'>{ICON_DOC}</div><div>"
        "<div class='pb-title'>PageBright</div>"
        "<div class='pb-sub'>Turn document photos and PDFs into clean, flat scans.</div></div></div>"
        f"<div class='pb-steps'>{items}</div>",
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        "<div class='pb-footer'>PageBright · Denish Kumbharvadiya (1AUA23BCS032) · "
        "Computer Vision and Applications Laboratory</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- helpers
def _is_local_session() -> bool:
    try:
        host = (st.context.headers.get("Host") or "").lower()
    except Exception:  # noqa: BLE001 - older Streamlit or bare mode
        return True
    return host.startswith(("localhost", "127.0.0.1", "[::1]", "0.0.0.0"))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_page_count(file_hash: str, _pdf_bytes: bytes) -> int:
    return get_pdf_page_count(_pdf_bytes)


@st.cache_data(show_spinner=False, max_entries=48)
def cached_pdf_page(file_hash: str, _pdf_bytes: bytes, page_index: int, dpi: int) -> np.ndarray:
    return render_pdf_page(_pdf_bytes, page_index, dpi)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_image(file_hash: str, _image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(_image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("This file could not be opened as an image. Try JPG or PNG.")
    return image


def source_page(source: dict, page_index: int) -> np.ndarray:
    if source["is_pdf"]:
        return cached_pdf_page(source["hash"], source["bytes"], page_index, source["dpi"])
    return cached_image(source["hash"], source["bytes"])


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def corner_editor(image: np.ndarray, corners_px: np.ndarray, key: str, version: int) -> np.ndarray:
    """Show draggable page corners on ``image``; return corners in image pixels."""
    height, width = image.shape[:2]
    scale = min(1.0, 1100.0 / max(height, width))
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    normalized = (np.asarray(corners_px, dtype=np.float32) / [max(1, width - 1), max(1, height - 1)]).round(5).tolist()
    value = _corner_editor(image=data_url, corners=normalized, version=version, max_height=620,
                           key=f"{key}_v{version}", default=normalized)
    return np.asarray(value, dtype=np.float32) * [width - 1, height - 1]


def process_page(image: np.ndarray, page_index: int, settings: dict, manual: np.ndarray | None = None) -> dict:
    """Run the pipeline on one page and return a page record with encoded outputs."""
    is_pdf, notice, needs_check = settings["is_pdf"], None, False
    kwargs = dict(mode=settings["mode"], min_area_fraction=settings["min_area"], denoise=settings["denoise"],
                  model_path=MODEL_PATH if MODEL_PATH.exists() else None)
    if manual is not None:
        images, report = process_image(image, manual_corners=manual, **kwargs)
    elif is_pdf and not settings["detect_pdf_edges"]:
        images, report = process_image(image, manual_corners=full_page_corners(image), trim_fraction=0, **kwargs)
        report["detection_method"] = "pdf_full_page"
    else:
        try:
            images, report = process_image(image, **kwargs)
        except ValueError:
            inset = 0.0 if is_pdf else 0.03
            images, report = process_image(image, manual_corners=full_page_corners(image, inset),
                                           trim_fraction=0, **kwargs)
            report["detection_method"] = "full_page_fallback" if is_pdf else "fallback"
            needs_check = True
            notice = (f"Page {page_index + 1}: the page edges couldn't be found automatically. "
                      "Drag the corners into place under **Adjust page** and press **Apply corners**.")
    report["page_number"] = page_index + 1
    return {
        "report": report,
        "files": {OUTPUT_FILENAMES[key]: encode_image(key, array) for key, array in images.items()},
        "rotation": 0, "edit_version": 0, "needs_check": needs_check, "notice": notice,
    }


def rotate_page_files(page: dict, quarter_turns: int) -> None:
    """Rotate every output of a page except the boundary preview."""
    rotate_code = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 3: cv2.ROTATE_90_CLOCKWISE}[quarter_turns % 4]
    for name, data in list(page["files"].items()):
        if name == OUTPUT_FILENAMES["boundary"]:
            continue
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        key = next(k for k, v in OUTPUT_FILENAMES.items() if v == name)
        page["files"][name] = encode_image(key, cv2.rotate(array, rotate_code))
    page["rotation"] = (page["rotation"] + quarter_turns) % 4


def build_pdf(result: dict) -> bytes:
    style = result["pdf_style"]
    order = [style] + [s for s in ("color", "gray", "bw") if s != style] + ["rectified"]
    pdf_images = []
    for page in result["pages"]:
        key = next(k for k in order if OUTPUT_FILENAMES[k] in page["files"])
        with Image.open(io.BytesIO(page["files"][OUTPUT_FILENAMES[key]])) as opened:
            pdf_images.append(opened.convert("1" if key == "bw" else "RGB"))  # 1-bit keeps B&W PDFs tiny
    first = pdf_images[0]
    if all(p["report"]["detection_method"] == "pdf_full_page" for p in result["pages"]):
        resolution = float(result["settings"]["dpi"])  # uncropped PDF pages keep their physical size
    else:
        resolution = max(72.0, first.width / 8.27)  # size photo scans like an A4 page
    buffer = io.BytesIO()
    first.save(buffer, format="PDF", save_all=True, append_images=pdf_images[1:], resolution=resolution)
    return buffer.getvalue()


def build_bundles(result: dict) -> None:
    """(Re)build the combined PDF and the ZIP after any change."""
    result["pdf_bytes"] = build_pdf(result)
    metadata = {
        "source_file": result["name"], "source_type": "pdf" if result["is_pdf"] else "image",
        "page_count": len(result["pages"]),
        "pdf_render_dpi": result["settings"]["dpi"] if result["is_pdf"] else None,
        "pdf_style": result["pdf_style"],
        "pages": [dict(p["report"], rotation_quarter_turns_ccw=p["rotation"]) for p in result["pages"]],
        "notices": [p["notice"] for p in result["pages"] if p["notice"]],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for page in result["pages"]:
            prefix = f"page_{page['report']['page_number']:03d}/"
            for name, data in page["files"].items():
                archive.writestr(prefix + name, data)
        archive.writestr("document_scan.pdf", result["pdf_bytes"])
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
    result["zip_bytes"] = buffer.getvalue()


def training_sample(image: np.ndarray, corners: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    ordered = order_corners(corners)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("Could not encode the labeled page.")
    label = {"image": "page.jpg", "width": int(width), "height": int(height),
             "corners_normalized": np.round(ordered / np.array([width, height], dtype=np.float32), 6).tolist()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("page.jpg", encoded.tobytes())
        archive.writestr("labels.json", json.dumps(label, indent=2))
    return buffer.getvalue()


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Scan settings")
    mode_label = st.selectbox(
        "Output styles", ("All styles", "Color only", "Grayscale only", "Black and white only"),
        help="All styles creates color, grayscale, and black-and-white versions of every page.",
    )
    mode = {"All styles": "all", "Color only": "color", "Grayscale only": "gray",
            "Black and white only": "bw"}[mode_label]
    if mode == "all":
        pdf_style = st.selectbox("Style for the combined PDF", ("color", "gray", "bw"),
                                 format_func=STYLE_LABELS.get, help="Black and white gives the smallest PDF.")
    else:
        pdf_style = mode
    denoise = st.toggle("Reduce image noise", value=True)
    min_area = st.slider("Smallest page size in photo", 5, 40, 8, 1, format="%d%%",
                         help="Raise this if detection picks a smaller object instead of the page.") / 100.0

# --------------------------------------------------------------------------- source file
uploaded = st.session_state.get("pb_uploader")
result = st.session_state.get("pb_result")
has_file = uploaded is not None or st.session_state.get("pb_demo", False)
header(2 if (result and has_file) else 1 if has_file else 0)

if _is_local_session():
    st.markdown("<p class='pb-privacy'>🔒 Runs on this computer. Your files are processed here and never uploaded "
                "anywhere.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='pb-privacy'>🔒 Files are processed in memory on this server, discarded when you close "
                "the page, and never used for training.</p>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a document photo or PDF", type=("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "pdf"),
    help="PDFs are processed page by page, up to 20 pages per file.", key="pb_uploader",
)
if uploaded is not None:
    st.session_state["pb_demo"] = False
    file_name, file_bytes = uploaded.name, uploaded.getvalue()
elif st.session_state.get("pb_demo") and DEMO_PATH.is_file():
    file_name, file_bytes = "sample_invoice.png", DEMO_PATH.read_bytes()
    left, right = st.columns([4, 1])
    left.caption("Using the built-in sample document.")
    if ui(right.button, "Clear sample", stretch=True):
        st.session_state["pb_demo"] = False
        st.session_state.pop("pb_result", None)
        st.rerun()
else:
    st.markdown("<div class='pb-cards'>" + "".join(
        f"<div class='pb-card'>{_icon(path)}<h4>{title}</h4><p>{text}</p></div>"
        for path, title, text in FEATURES) + "</div>", unsafe_allow_html=True)
    if DEMO_PATH.is_file() and ui(st.button, "Try it with a sample document", type="primary",
                                  icon=":material/description:"):
        st.session_state["pb_demo"] = True
        st.rerun()
    footer()
    st.stop()

is_pdf = file_name.lower().endswith(".pdf")
file_hash = hashlib.sha256(file_bytes).hexdigest()

pdf_dpi, detect_pdf_edges = 144, False
if is_pdf:
    with st.sidebar:
        st.subheader("PDF options")
        pdf_dpi = st.slider("Render quality (DPI)", 96, 200, 144, 12,
                            help="Higher values keep small text sharper but use more memory.")
        detect_pdf_edges = st.checkbox(
            "Pages are photos of paper", value=False,
            help="Turn on when each PDF page is a photo of a sheet of paper that needs cropping. "
                 "Leave off for normal scanned or digital PDFs.",
        )
if MODEL_PATH.exists():
    st.sidebar.caption("Using your trained corner model for page detection.")

source = {"hash": file_hash, "bytes": file_bytes, "is_pdf": is_pdf, "dpi": pdf_dpi}
try:
    page_count = cached_page_count(file_hash, file_bytes) if is_pdf else 1
    first_page = source_page(source, 0)
except (ValueError, OSError, cv2.error) as error:
    st.error(str(error))
    st.stop()

settings = {
    "file": file_hash, "is_pdf": is_pdf, "mode": mode, "denoise": denoise, "min_area": min_area,
    "dpi": pdf_dpi if is_pdf else None, "detect_pdf_edges": detect_pdf_edges if is_pdf else None,
}
settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
if result and result["file"] != file_hash:
    st.session_state.pop("pb_result", None)
    result = None

h, w = first_page.shape[:2]
st.caption(f"{file_name} · {page_count} page{'s' if page_count != 1 else ''} · {w:,} × {h:,} px"
           + (f" · {pdf_dpi} DPI" if is_pdf else ""))

if result is None:
    preview_col, action_col = ui(st.columns, [1.2, 1], gap="large")
    with preview_col:
        preview_index = 0
        if is_pdf and page_count > 1:
            preview_index = st.selectbox("Preview page", range(page_count),
                                         format_func=lambda i: f"Page {i + 1} of {page_count}",
                                         help="Changes the preview only. Every page is enhanced.")
        ui(st.image, to_rgb(source_page(source, preview_index)), stretch=True,
           caption=f"Page {preview_index + 1}" if is_pdf else "Original photo")
    with action_col:
        st.subheader("Ready to scan")
        st.markdown(
            f"PageBright will enhance all **{page_count} pages** and combine them into one PDF."
            if is_pdf else
            "PageBright will find the page, flatten it, and clean up the lighting. "
            "You can fine-tune the corners afterwards."
        )
        process_clicked = ui(st.button, f"Enhance {page_count} pages" if is_pdf and page_count > 1
                             else "Enhance document", type="primary", stretch=True,
                             icon=":material/auto_awesome:")
else:
    process_clicked = False
    if result["settings_hash"] != settings_hash:
        warn_col, button_col = ui(st.columns, [3, 1], vertical_alignment="center")
        warn_col.warning("You changed the scan settings. Re-run to apply them to every page.")
        process_clicked = ui(button_col.button, "Re-run with new settings", type="primary", stretch=True)

# --------------------------------------------------------------------------- processing
if process_clicked:
    pages = []
    progress = st.progress(0.0, text="Starting…")
    try:
        for index in range(page_count):
            progress.progress(index / page_count, text=f"Enhancing page {index + 1} of {page_count}…")
            pages.append(process_page(source_page(source, index), index, settings))
        progress.progress(1.0, text="Building PDF…")
        result = {"file": file_hash, "name": Path(file_name).name, "is_pdf": is_pdf, "settings": settings,
                  "settings_hash": settings_hash, "pdf_style": pdf_style, "pages": pages}
        build_bundles(result)
        st.session_state["pb_result"] = result
        st.session_state.pop("pb_page_pills", None)
        st.session_state["pb_selected_page"] = next((i for i, p in enumerate(pages) if p["needs_check"]), 0)
    except (OSError, ValueError, cv2.error, RuntimeError) as error:
        st.error(f"Something went wrong while scanning: {error}")
        result = None
    finally:
        progress.empty()
    if result is not None:
        st.rerun()

if result is None:
    footer()
    st.stop()

# --------------------------------------------------------------------------- results
if result["pdf_style"] != pdf_style and OUTPUT_FILENAMES.get(pdf_style) in result["pages"][0]["files"]:
    result["pdf_style"] = pdf_style
    build_bundles(result)

pages = result["pages"]
stem = Path(result["name"]).stem
auto_count = sum(1 for p in pages if p["report"]["detection_method"] in AUTO_METHODS)
manual_count = sum(1 for p in pages if p["report"]["detection_method"] == "manual")
check_count = sum(1 for p in pages if p["needs_check"])

st.subheader("Your scan is ready")
m1, m2, m3 = st.columns(3)
m1.metric("Pages", len(pages))
if result["is_pdf"] and not result["settings"]["detect_pdf_edges"] and manual_count == 0:
    m2.metric("Page edges", "Full PDF pages")
elif manual_count == 0:
    m2.metric("Page edges", "Found ✓" if auto_count == len(pages) else f"{auto_count} of {len(pages)} found")
elif len(pages) == 1:
    m2.metric("Page edges", "Adjusted by you")
else:
    m2.metric("Page edges", f"{auto_count} auto · {manual_count} by you")
m3.metric("Pages to check", check_count if check_count else "None ✓")
for page in pages:
    if page["needs_check"] and page["notice"]:
        st.warning(page["notice"])

d1, d2 = st.columns(2)
with d1:
    ui(st.download_button, f"Download combined PDF ({STYLE_LABELS.get(result['pdf_style'], 'Color').lower()})",
       result["pdf_bytes"], file_name=f"{stem}_pagebright.pdf", mime="application/pdf", type="primary",
       stretch=True, icon=":material/picture_as_pdf:", key="dl_pdf")
with d2:
    ui(st.download_button, "Download every page + details (ZIP)", result["zip_bytes"],
       file_name=f"{stem}_pagebright_results.zip", mime="application/zip", stretch=True,
       icon=":material/folder_zip:", key="dl_zip")

selected = 0
if len(pages) > 1:
    st.session_state.setdefault("pb_selected_page", 0)
    st.session_state["pb_selected_page"] = min(st.session_state["pb_selected_page"], len(pages) - 1)
    label = lambda i: f"{i + 1} ⚠" if pages[i]["needs_check"] else str(i + 1)  # noqa: E731
    if hasattr(st, "pills"):
        picked = st.pills("Page", list(range(len(pages))), format_func=label, key="pb_page_pills",
                          default=st.session_state["pb_selected_page"])
        if picked is not None:
            st.session_state["pb_selected_page"] = picked
    else:
        st.session_state["pb_selected_page"] = st.selectbox("Page", range(len(pages)),
                                                            format_func=lambda i: f"Page {label(i)}",
                                                            index=st.session_state["pb_selected_page"])
    selected = st.session_state["pb_selected_page"]
page = pages[selected]
report = page["report"]
original = source_page(source, selected)

out_col, edit_col = ui(st.columns, [1, 1], gap="large")
with out_col:
    st.markdown(f"<p class='pb-label'>Result{f' · page {selected + 1}' if len(pages) > 1 else ''}</p>",
                unsafe_allow_html=True)
    style_keys = [k for k in ("color", "gray", "bw") if OUTPUT_FILENAMES[k] in page["files"]]
    tabs = st.tabs([STYLE_LABELS[k] for k in style_keys] + ["Details"])
    for tab, key in zip(tabs, style_keys):
        with tab:
            name = OUTPUT_FILENAMES[key]
            ui(st.image, page["files"][name], stretch=True)
            ui(st.download_button, f"Download this page ({Path(name).suffix[1:].upper()})", page["files"][name],
               file_name=f"{stem}_page_{selected + 1:03d}_{key}{Path(name).suffix}",
               mime="image/png" if name.endswith(".png") else "image/jpeg",
               key=f"dl_{selected}_{key}", icon=":material/download:")
    with tabs[-1]:
        st.markdown(
            f"- **Page edges:** {METHOD_LABELS.get(report['detection_method'], report['detection_method'])}"
            f" (`{report['detection_method']}`)\n"
            f"- **Output size:** {report['rectified_size_px']['width']} × {report['rectified_size_px']['height']} px\n"
            f"- **Processing time:** {report['processing_seconds']:.2f} s"
        )
        ui(st.image, page["files"][OUTPUT_FILENAMES["boundary"]], caption="Page boundary used", stretch=True)
        with st.expander("Raw metadata"):
            st.json(report)

with edit_col:
    st.markdown("<p class='pb-label'>Adjust page</p>", unsafe_allow_html=True)
    corners_now = corner_editor(original, np.asarray(report["corners_xy"], dtype=np.float32),
                                key=f"edit_{file_hash[:12]}_{selected}", version=page["edit_version"])
    b1, b2 = st.columns(2)
    apply_clicked = ui(b1.button, "Apply corners", type="primary", stretch=True, key=f"apply_{selected}",
                       icon=":material/check:")
    redetect_clicked = ui(b2.button, "Detect automatically", stretch=True, key=f"redetect_{selected}",
                          icon=":material/auto_fix_high:")
    r1, r2 = st.columns(2)
    rotate_left = ui(r1.button, "Rotate left", stretch=True, key=f"rotl_{selected}", icon=":material/rotate_left:")
    rotate_right = ui(r2.button, "Rotate right", stretch=True, key=f"rotr_{selected}",
                      icon=":material/rotate_right:")

    if apply_clicked or redetect_clicked:
        try:
            run_settings = dict(result["settings"], detect_pdf_edges=True) if redetect_clicked else result["settings"]
            new_page = process_page(original, selected, run_settings, None if redetect_clicked else corners_now)
            if redetect_clicked and new_page["needs_check"]:
                st.toast("Couldn't find the page edges automatically. Drag the corners instead.", icon="⚠️")
            new_page["edit_version"] = page["edit_version"] + 1
            if page["rotation"]:
                rotate_page_files(new_page, page["rotation"])
            pages[selected] = new_page
            build_bundles(result)
            st.rerun()
        except (OSError, ValueError, cv2.error, RuntimeError) as error:
            st.error(f"Could not update page {selected + 1}: {error}")
    if rotate_left or rotate_right:
        rotate_page_files(page, 1 if rotate_left else 3)
        build_bundles(result)
        st.rerun()

    if report["detection_method"] == "manual":
        st.caption("Help train the corner model: save this page with your corner positions as a labeled "
                   "example (stays on your device; see docs/TRAINING_GUIDE.md).")
        st.download_button("Download labeled sample", training_sample(original, np.asarray(report["corners_xy"])),
                           file_name=f"pagebright_sample_{stem}_p{selected + 1}.zip", mime="application/zip",
                           key=f"train_{selected}")

footer()

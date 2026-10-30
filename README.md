# PageBright

PageBright enhances document photos and multi-page PDFs. It can straighten photographed pages, improve illumination and contrast, and export enhanced images, a combined PDF, and a ZIP of per-page results.

## Use the app

Upload a photo or PDF (or choose **Try it with a sample document**), pick output styles, and select **Enhance**. PDFs are processed page by page; download one combined PDF or all outputs as a ZIP. For ordinary scanned PDFs, leave photographed-page edge detection off. Turn it on when a PDF page contains a photo of a sheet that should be cropped.

## Fixing a page

After a scan, each page's result is shown next to an **Adjust page** panel. Drag the four round handles onto the paper's corners (a magnifier appears while dragging; touch works too), then choose **Apply corners**. Only that page is reprocessed and the combined PDF and ZIP update. **Detect automatically** re-runs edge detection, and **Rotate left/right** turn the page. Pages where no edge was found are flagged instead of stopping the scan.

## Enhancement

The paper background is estimated with a max filter and median blur, then divided out, giving a white page without shadows, paper tint, or halos around text. Black and white combines adaptive and global thresholds and rejoins faint table lines. Black-and-white PDFs use 1-bit images and stay small.

## Inputs and limits

- Photo formats: JPG, PNG, WEBP, BMP, and TIFF.
- PDFs: up to 20 pages per file; choose 96–200 DPI.
- Files are processed in memory on the hosting server, discarded after the session, and not added to a shared training set.

## Detection and training

Automatic detection uses several edge, threshold, and contour methods. If a trained model is placed at models/pagebright_corners.onnx, PageBright will try that model first. Pages you adjust by hand can be exported as labeled training examples. See TRAINING_GUIDE.md for the optional offline training workflow.

## Deploy on Streamlit Community Cloud

The app file is app.py at the root of this repository. Select the main branch and app.py in the Community Cloud dashboard. Dependencies are listed in requirements.txt.

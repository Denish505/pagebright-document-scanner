# PageBright

PageBright enhances document photos and multi-page PDFs. It can straighten photographed pages, improve illumination and contrast, and export enhanced images, a combined PDF, and a ZIP of per-page results.

## Use the app

Upload a photo or PDF, choose output styles, and select **Enhance**. PDFs are processed page by page; download one combined PDF or all outputs as a ZIP. For ordinary scanned PDFs, leave photographed-page edge detection off. Turn it on when a PDF page contains a photo of a sheet that should be cropped.

## Inputs and limits

- Photo formats: JPG, PNG, WEBP, BMP, and TIFF.
- PDFs: up to 20 pages per file; choose 96–200 DPI.
- Files are processed on the hosting server and are not added to a shared training set. Avoid sensitive documents.

## Detection and training

Automatic detection uses several edge, threshold, and contour methods. If a trained model is placed at models/pagebright_corners.onnx, PageBright will try that model first. Manual corners can be exported as labeled training examples. See TRAINING_GUIDE.md for the optional offline training workflow.

## Deploy on Streamlit Community Cloud

The app file is app.py at the root of this repository. Select the main branch and app.py in the Community Cloud dashboard. Dependencies are listed in requirements.txt.

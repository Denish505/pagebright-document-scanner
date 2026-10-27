# PageBright model training

PageBright's default detector uses OpenCV edges and contours. It is not a neural network and cannot learn from unlabeled uploads. The hosted app does not add user files to a shared training set.

## Export labeled examples

1. Upload a document photo or PDF page.
2. Open **Manual page corners** and enter accurate top-left, top-right, bottom-right, and bottom-left coordinates.
3. Process the page, then choose **Prepare labeled training sample**.
4. Save each downloaded ZIP in one training folder.
5. Collect at least 30 varied examples; 100–500 accurately labeled pages is a more useful starting point.

Use different backgrounds, paper colors, lighting, camera angles, and shadows. Do not use private documents unless you are permitted to use them for training.

## Train offline

Install the optional packages from requirements-training.txt, then run:

    python scripts/train_page_model.py --data-dir training_data --epochs 20

The script fine-tunes an ImageNet-initialized MobileNetV3 Small corner regressor and exports models/pagebright_corners.onnx. The app tries this model first and falls back to contour detection if the model is absent or returns invalid corners.

Training quality depends on label accuracy and dataset variety. Review validation loss and manually inspect scans before relying on a newly trained model. Training requires internet access for pretrained weights unless --no-pretrained is used.

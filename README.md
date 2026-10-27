# PageBright cloud deployment

This folder contains only the PageBright application source needed for a Streamlit Community Cloud deployment. Student proposal documents and generated reports are intentionally excluded.

## Deploy

1. Create a GitHub repository and add the contents of this folder to its root.
2. Connect GitHub at https://share.streamlit.io/.
3. Choose **Create app**, select the repository and branch, and set the app file to `app.py`.
4. Deploy. Streamlit installs the packages in `requirements.txt`.

The hosted app processes uploaded images on the hosting server. It uses a temporary working directory that is deleted after processing. Do not upload sensitive documents. Local use remains available from the original project folder with `Start_PageBright.bat`.

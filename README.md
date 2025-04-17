# Real vs. AI Image Classifier

Web app to classify images as "Real" or "AI-generated" using a PyTorch model.

## Technologies

* Python
* Streamlit
* PyTorch

## Setup

1.  Clone repo.
2.  `cd <repository_name>`
3.  Create virtual environment (`python -m venv venv`, activate).
4.  `pip install -r requirements.txt`
5.  `streamlit run realvsaifrontend.py`

## Streamlit Cloud Deployment

* Push to GitHub.
* Create new app on Streamlit Cloud, linking to `realvsaifrontend.py`.
* Include `packages.txt` (with `libz-dev`, `libjpeg-dev`) and `runtime.txt` (with `python-3.12`).

## Usage

1.  Upload image in the app.
2.  Get "Real" or "AI Generated" prediction.

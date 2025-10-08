# PH421 - Molecular Property Predictor

This project is part of the PH421 course assignment.  
It implements a molecular property prediction pipeline using machine learning models trained on chemical datasets.  
A Streamlit app is provided for an interactive interface.

---

## Features
- **Aqueous Solubility (ESOL)** prediction
- **Lipophilicity (LogP)** prediction
- **Blood-Brain Barrier Permeability (BBBP)** classification
- **Toxicity** multi-label classification (12 endpoints)

---
---

## Models
The trained models are **not included in this repository** (to avoid large file size issues).  
Please download them from Google Drive:  

👉 [Download Models](https://drive.google.com/drive/folders/1GExedUoRx8k84HtIRk8QLAu2kDryq4ye?usp=drive_link)

After downloading, place them in src/models

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/ARYAN0201/PH421.git
   cd PH421
   ```
2. Create a virtual environment:
   ```bash
    python3 -m venv .phenv
    source .phenv/bin/activate
    .phenv\Scripts\activate
    ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run using streamlit:
   ```bash
   streamlit run app.py
   ```
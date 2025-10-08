import streamlit as st
from pathlib import Path
from src.inference import InferencePipeline

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Molecular Property Predictor",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Molecular Property Predictor")
st.markdown(
    """
    Enter a **SMILES string** below to predict molecular properties:
    - **Aqueous solubility** (ESOL)
    - **Lipophilicity**
    - **BBB Permeability**
    - **Toxicity Profile**
    """
)

# -------------------------------
# Sidebar for input
# -------------------------------
st.sidebar.header("Input SMILES")
smiles_input = st.sidebar.text_area("Enter SMILES string here:")

# -------------------------------
# Load model pipeline
# -------------------------------
@st.cache_resource
def load_pipeline():
    model_paths = {
        "tox": Path("./src/models/toxicity_rf.pkl"),
        "esol": Path("./src/models/esol_lgb.txt"),
        "lipophilicity": Path("./src/models/klipo_xgb.pkl"),
        "bbbp": Path("./src/models/bbbp_extratrees.pkl")
    }
    return InferencePipeline(model_paths)

pipeline = load_pipeline()

# -------------------------------
# Toxicity Labels
# -------------------------------
tox_labels = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# -------------------------------
# Predict Button
# -------------------------------
if st.sidebar.button("Predict"):
    if not smiles_input.strip():
        st.warning("Please enter a valid SMILES string!")
    else:
        with st.spinner("Running predictions..."):
            results = pipeline.predict(smiles_input.strip())

        st.success("✅ Prediction Complete!")

        # -------------------------------
        # Display Results in Columns
        # -------------------------------
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Aqueous Solubility (ESOL)", value=f"{results['esol']:.4f}")

        with col2:
            st.metric(label="Lipophilicity", value=f"{results['lipophilicity']:.4f}")

        with col3:
            if results['bbbp'] == 1:
                st.success("BBB Permeable ✅")
            else:
                st.error("BBB Not Permeable ❌")

        with col4:
            tox_pred = results['tox']
            tox_indices = [i for i, v in enumerate(tox_pred) if v == 1]
            if tox_indices:
                tox_output = ", ".join([tox_labels[i] for i in tox_indices])
                st.error(f"Toxic: {tox_output}")
            else:
                st.success("Non-toxic ✅")

        # -------------------------------
        # Raw Predictions
        # -------------------------------
        with st.expander("Show raw predictions"):
            st.json(results)

st.markdown("---")
st.markdown(
    """
    **Instructions:**  
    1. Enter a valid SMILES string in the sidebar.  
    2. Click **Predict**.  
    3. Results are displayed with colored indicators for toxicity and BBB permeability.  
    """
)
"""
Laptop Price Prediction - Student Demo App
------------------------------------------
A small Streamlit web app that lets students interact with the trained
model from the "Laptop Price Prediction" notebook (practice3.ipynb),
without needing to run Jupyter.

It loads the same laptop_price_model_full_dataset.pkl produced by that
notebook (a pipeline that already contains the imputer + scaler + one-hot
encoder + Random Forest), so the app sends raw feature values and the
pipeline does all the encoding itself.

The drop-down lists are built from laptop_price.csv, using the same
feature extraction as the notebook, so every choice offered to the user
is something the model has actually seen.

Files that must be in the SAME folder as this script:
    laptop_app.py
    laptop_price_model_full_dataset.pkl
    laptop_price.csv

To run:
    pip install streamlit
    streamlit run laptop_app.py

Use the same scikit-learn version that saved the model, otherwise loading
the .pkl file may fail or give warnings.

NOTE: This is an educational demo. Prices reflect the dataset, not today's
market.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Cross-validation RMSE of the best model (best_mean_rmse in the notebook).
# Set to None to hide the "typical error" line.
TYPICAL_ERROR_EUROS = 279

HERE = Path(__file__).parent
MODEL_PATH = HERE / "laptop_price_model_full_dataset.pkl"
DATA_PATH = HERE / "laptop_price.csv"

# Same feature lists, in the same order, as the notebook
NUMERICAL_FEATURES = ["Inches", "Touchscreen", "Ram_GB", "Cpu_GHz", "IPS", "Res_W", "Res_H",
                      "Flash_GB", "Hybrid_GB", "PPI", "SSD_GB", "HDD_GB"]
CATEGORICAL_FEATURES = ["Company", "Product", "TypeName", "Cpu_family", "Gpu_brand", "OpSys"]

# Friendly names for common resolutions (others are shown as WIDTH x HEIGHT)
RESOLUTION_NAMES = {
    (1366, 768): "HD",
    (1600, 900): "HD+",
    (1920, 1080): "Full HD",
    (2560, 1440): "Quad HD",
    (3200, 1800): "QHD+",
    (3840, 2160): "4K Ultra HD",
}

# ---------------------------------------------------------------------------
# Load the trained model and the data used for the drop-down lists
# ---------------------------------------------------------------------------

for required_file in (MODEL_PATH, DATA_PATH):
    if not required_file.exists():
        st.error(f"File not found: `{required_file.name}`. Put it in the same folder as this script.")
        st.stop()


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def parse_storage(s):
    """Same function as the notebook: '256GB SSD + 1TB HDD' -> GB per storage type."""
    out = {"SSD_GB": 0.0, "HDD_GB": 0.0, "Hybrid_GB": 0.0, "Flash_GB": 0.0}
    names = {"SSD": "SSD_GB", "HDD": "HDD_GB", "Hybrid": "Hybrid_GB", "Flash Storage": "Flash_GB"}
    for size, unit, kind in re.findall(r"(\d+(?:\.\d+)?)\s*(GB|TB)\s*(SSD|HDD|Hybrid|Flash Storage)", s):
        out[names[kind]] += float(size) * (1024 if unit == "TB" else 1)
    return pd.Series(out)


def cpu_family(s):
    """Same function as the notebook."""
    for pat in ["Core i7", "Core i5", "Core i3", "Core M", "Celeron", "Pentium",
                "Atom", "Xeon", "Ryzen", "A12", "A10", "A9", "A8", "A6", "E-Series", "FX"]:
        if pat in s:
            return pat
    return "Other"


@st.cache_data
def load_catalogue():
    """Reads the CSV and extracts the same features as the notebook."""
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    df["Ram_GB"] = df["Ram"].str.replace("GB", "").astype(int)
    df = pd.concat([df, df["Memory"].apply(parse_storage)], axis=1)

    res = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)").astype(float)
    df["Res_W"], df["Res_H"] = res[0].astype(int), res[1].astype(int)
    df["Touchscreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)
    df["IPS"] = df["ScreenResolution"].str.contains("IPS").astype(int)

    df["Cpu_GHz"] = df["Cpu"].str.extract(r"(\d+(?:\.\d+)?)\s*GHz").astype(float)
    df["Cpu_brand"] = df["Cpu"].str.split().str[0]      # only used to narrow the CPU list
    df["Cpu_family"] = df["Cpu"].apply(cpu_family)
    df["Gpu_brand"] = df["Gpu"].str.split().str[0]
    return df


model = load_model()
catalogue = load_catalogue()

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def pick(options, preferred):
    """Index of `preferred` in `options`; 0 if it is not there."""
    return options.index(preferred) if preferred in options else 0


def storage_label(gb):
    gb = float(gb)
    if gb == 0:
        return "None"
    if gb >= 1024:
        tb = gb / 1024
        return f"{tb:g} TB"
    return f"{gb:g} GB"


def resolution_label(res):
    width, height = res
    name = RESOLUTION_NAMES.get((width, height))
    return f"{width} x {height}" + (f"  ({name})" if name else "")


def storage_options(column):
    """Sizes seen in the data for this storage type; 'None' (0) is always first."""
    return sorted(set(catalogue[column].tolist()) | {0.0})


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("💻 Laptop Price Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Laptop Price Prediction** notebook.
    Choose a laptop's specifications below and the trained model will estimate its
    price in euros. Picking a model fills in its usual specifications, which you can
    then change.
    """
)
st.divider()

# ---- 1. Brand and model ---------------------------------------------------
st.subheader("1. Brand and model")
col1, col2 = st.columns(2)

with col1:
    companies = sorted(catalogue["Company"].unique().tolist())
    company = st.selectbox("Laptop Company", companies)

company_rows = catalogue[catalogue["Company"] == company]

with col2:
    products = sorted(company_rows["Product"].unique().tolist())
    product = st.selectbox(f"Model ({len(products)} available)", products, key=f"product|{company}")

# The usual configuration of the chosen model, used as the starting values below
typical = catalogue[(catalogue["Company"] == company) & (catalogue["Product"] == product)].mode().iloc[0]

with col1:
    types = sorted(catalogue["TypeName"].unique().tolist())
    type_name = st.selectbox("Laptop Type", types, index=pick(types, typical["TypeName"]),
                             key=f"type|{product}")

with col2:
    op_systems = sorted(catalogue["OpSys"].unique().tolist())
    op_sys = st.selectbox("Operating System", op_systems, index=pick(op_systems, typical["OpSys"]),
                          key=f"os|{product}")

# ---- 2. Processor and graphics -------------------------------------------
st.subheader("2. Processor and graphics")
col1, col2 = st.columns(2)

with col1:
    cpu_brands = sorted(catalogue["Cpu_brand"].unique().tolist())
    cpu_brand = st.selectbox("CPU Company", cpu_brands, index=pick(cpu_brands, typical["Cpu_brand"]),
                             key=f"cpu_brand|{product}")

brand_rows = catalogue[catalogue["Cpu_brand"] == cpu_brand]

with col2:
    cpu_families = sorted(brand_rows["Cpu_family"].unique().tolist())
    cpu_fam = st.selectbox("CPU Family", cpu_families, index=pick(cpu_families, typical["Cpu_family"]),
                           key=f"cpu_family|{product}|{cpu_brand}")

with col1:
    speeds = sorted(brand_rows[brand_rows["Cpu_family"] == cpu_fam]["Cpu_GHz"].unique().tolist())
    cpu_ghz = st.selectbox("CPU Speed (GHz)", speeds, index=pick(speeds, float(typical["Cpu_GHz"])),
                           key=f"cpu_ghz|{product}|{cpu_brand}|{cpu_fam}")

with col2:
    gpu_brands = sorted(catalogue["Gpu_brand"].unique().tolist())
    gpu_brand = st.selectbox("Graphics Card Company", gpu_brands,
                             index=pick(gpu_brands, typical["Gpu_brand"]), key=f"gpu|{product}")

# ---- 3. Memory and storage -----------------------------------------------
st.subheader("3. Memory and storage")
col1, col2 = st.columns(2)

with col1:
    ram_options = sorted(catalogue["Ram_GB"].unique().tolist())
    ram = st.selectbox("RAM", ram_options, index=pick(ram_options, int(typical["Ram_GB"])),
                       format_func=lambda v: f"{v} GB", key=f"ram|{product}")

    ssd_options = storage_options("SSD_GB")
    ssd = st.selectbox("SSD Storage", ssd_options, index=pick(ssd_options, float(typical["SSD_GB"])),
                       format_func=storage_label, key=f"ssd|{product}")

    flash_options = storage_options("Flash_GB")
    flash = st.selectbox("Flash Storage", flash_options, index=pick(flash_options, float(typical["Flash_GB"])),
                         format_func=storage_label, key=f"flash|{product}")

with col2:
    hdd_options = storage_options("HDD_GB")
    hdd = st.selectbox("Hard Disk (HDD) Storage", hdd_options, index=pick(hdd_options, float(typical["HDD_GB"])),
                       format_func=storage_label, key=f"hdd|{product}")

    hybrid_options = storage_options("Hybrid_GB")
    hybrid = st.selectbox("Hybrid (SSHD) Storage", hybrid_options,
                          index=pick(hybrid_options, float(typical["Hybrid_GB"])),
                          format_func=storage_label, key=f"hybrid|{product}")

# ---- 4. Screen -----------------------------------------------------------
st.subheader("4. Screen")
col1, col2 = st.columns(2)

with col1:
    size_options = sorted(catalogue["Inches"].unique().tolist())
    inches = st.selectbox("Screen Size", size_options, index=pick(size_options, float(typical["Inches"])),
                          format_func=lambda v: f'{v:g}"', key=f"inches|{product}")

    panel_options = ["Standard panel", "IPS panel"]
    panel = st.selectbox("Screen Type", panel_options, index=int(typical["IPS"]), key=f"panel|{product}")

with col2:
    resolutions = sorted(
        {(int(w), int(h)) for w, h in zip(catalogue["Res_W"], catalogue["Res_H"])},
        key=lambda r: r[0] * r[1],
    )
    resolution = st.selectbox(
        "Resolution", resolutions,
        index=pick(resolutions, (int(typical["Res_W"]), int(typical["Res_H"]))),
        format_func=resolution_label, key=f"res|{product}",
    )

    touch = st.radio("Touchscreen", ["No", "Yes"], index=int(typical["Touchscreen"]),
                     horizontal=True, key=f"touch|{product}")

st.divider()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if st.button("Predict Price", type="primary", use_container_width=True):
    res_w, res_h = resolution

    user_input = pd.DataFrame({
        "Inches": [float(inches)],
        "Touchscreen": [1 if touch == "Yes" else 0],
        "Ram_GB": [int(ram)],
        "Cpu_GHz": [float(cpu_ghz)],
        "IPS": [1 if panel == "IPS panel" else 0],
        "Res_W": [float(res_w)],
        "Res_H": [float(res_h)],
        "Flash_GB": [float(flash)],
        "Hybrid_GB": [float(hybrid)],
        "PPI": [float(np.sqrt(res_w ** 2 + res_h ** 2) / inches)],   # same formula as the notebook
        "SSD_GB": [float(ssd)],
        "HDD_GB": [float(hdd)],
        "Company": [company],
        "Product": [product],
        "TypeName": [type_name],
        "Cpu_family": [cpu_fam],
        "Gpu_brand": [gpu_brand],
        "OpSys": [op_sys],
    })[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    predicted_price = float(model.predict(user_input)[0])

    st.success(f"### Estimated Price: €{predicted_price:,.0f}")
    if TYPICAL_ERROR_EUROS is not None:
        st.caption(f"Typical error of this model (cross-validation RMSE): about €{TYPICAL_ERROR_EUROS:,.0f}")

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(user_input.T.rename(columns={0: "Value"}).astype(str))

st.divider()
st.caption(
    "Built for teaching purposes from a laptop price dataset (prices in euros). "
    "The estimate reflects the data the model was trained on, not today's market prices."
)

"""
Laptop Price Prediction - Student Demo App  (version 3)
-------------------------------------------------------
A small Streamlit web app that lets students interact with the trained
model from the "Laptop Price Prediction" notebook (practice3.ipynb),
without needing to run Jupyter.

It loads the same laptop_price_model_full_dataset.pkl produced by that
notebook (a pipeline that already contains the imputer + scaler + one-hot
encoder + Random Forest), so the app sends raw feature values and the
pipeline does all the encoding itself.

HOW THE DROP-DOWN LISTS WORK
The lists are built from laptop_price.csv. To stop impossible laptops from
being selected without being too restrictive, the choices are limited in
three levels:

  1. Fixed by the MODEL      laptop type, screen size, resolution
                             (a 13.3" MacBook Pro only offers 2560 x 1600,
                              and its screen size can only be 13.3" or 15.4")
  2. Limited by the BRAND    touchscreen, screen type, operating system,
                             CPU company / family / speed, graphics company
                             (no touchscreen or Windows on a MacBook)
  3. Limited by the TYPE     RAM and every storage type
                             (any amount seen in laptops of the same type,
                              from any brand)

A list with only one possible value is shown greyed-out. Starting values are
the usual configuration of the chosen model, and every list can be changed.

Files that must be in the SAME folder as this script:
    laptop_app.py
    laptop_price_model_full_dataset.pkl
    laptop_price.csv
    requirements.txt   (only needed for installing / deploying)

To run:
    pip install -r requirements.txt
    streamlit run laptop_app.py

Only load .pkl files you created yourself: unpickling a file from an
unknown source can run arbitrary code.

NOTE: This is an educational demo. Prices reflect the dataset, not today's
market.
"""

import pickle
import re
import warnings
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

# Columns that must exist in laptop_price.csv
REQUIRED_COLUMNS = ["Company", "Product", "TypeName", "Inches", "ScreenResolution",
                    "Cpu", "Ram", "Memory", "Gpu", "OpSys"]

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
def load_model(file_mtime):
    """Loads the pickle. Returns (model, list of warning messages) or (None, [error])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except Exception as error:      # wrong scikit-learn version, corrupt file, ...
            return None, [f"{type(error).__name__}: {error}"]
    return model, [str(w.message) for w in caught]


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
def load_catalogue(file_mtime):
    """Reads the CSV and extracts the same features as the notebook.
    Returns (catalogue, number of rows that had to be skipped)."""
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, missing

    df["Ram_GB"] = pd.to_numeric(df["Ram"].str.extract(r"(\d+)")[0], errors="coerce")
    df = pd.concat([df, df["Memory"].apply(parse_storage)], axis=1)

    res = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)").astype(float)
    df["Res_W"], df["Res_H"] = res[0], res[1]
    df["Touchscreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)
    df["IPS"] = df["ScreenResolution"].str.contains("IPS").astype(int)

    df["Cpu_GHz"] = df["Cpu"].str.extract(r"(\d+(?:\.\d+)?)\s*GHz").astype(float)
    df["Cpu_brand"] = df["Cpu"].str.split().str[0]      # only used to narrow the CPU list
    df["Cpu_family"] = df["Cpu"].apply(cpu_family)
    df["Gpu_brand"] = df["Gpu"].str.split().str[0]

    # Rows where a number could not be read would put NaN into the drop-downs
    usable = df[["Ram_GB", "Res_W", "Res_H", "Cpu_GHz", "Inches"]].notna().all(axis=1)
    skipped = int((~usable).sum())
    df = df[usable].copy()

    df["Ram_GB"] = df["Ram_GB"].astype(int)
    df["Res_W"], df["Res_H"] = df["Res_W"].astype(int), df["Res_H"].astype(int)
    df["Resolution"] = df["Res_W"].astype(str) + "x" + df["Res_H"].astype(str)   # one value per resolution
    return df.reset_index(drop=True), skipped


model, model_messages = load_model(MODEL_PATH.stat().st_mtime)
if model is None:
    st.error(
        "The model file could not be loaded. This usually means the scikit-learn version here "
        "is different from the one that saved the model.\n\n"
        f"`{model_messages[0]}`\n\nInstall the same version (see requirements.txt) or re-run the "
        "notebook to save the model again."
    )
    st.stop()

catalogue, skipped_rows = load_catalogue(DATA_PATH.stat().st_mtime)
if catalogue is None:
    st.error(f"`laptop_price.csv` is missing these columns: {', '.join(skipped_rows)}")
    st.stop()

# Make sure the model expects exactly the columns this app sends
expected_columns = getattr(model, "feature_names_in_", None)
if expected_columns is not None and set(expected_columns) != set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES):
    st.error("The model was trained on different features than this app sends. "
             f"Model expects: {list(expected_columns)}")
    st.stop()

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def storage_label(gb):
    gb = float(gb)
    if gb == 0:
        return "None"
    if gb >= 1024:
        return f"{gb / 1024:g} TB"
    return f"{gb:g} GB"


def resolution_label(key):
    width, height = (int(v) for v in key.split("x"))
    name = RESOLUTION_NAMES.get((width, height))
    return f"{width} x {height}" + (f"  ({name})" if name else "")


def resolution_size(key):
    width, height = (int(v) for v in key.split("x"))
    return width * height


def choose(label, column, rows, path, *, format_func=str, sort_key=None,
           default_rows=None, append_to_path=True):
    """
    Draws one drop-down whose options are ONLY the values of `column` that exist in `rows`.

    rows           the laptops that are still possible given the choices made so far
    path           list of the choices made so far; it is part of the widget key, so when an
                   earlier choice changes, this drop-down is rebuilt with a fresh default
    default_rows   if given, the starting value is the most common value in these rows
    append_to_path add this choice to `path` (so later drop-downs depend on it)

    Returns (selected value, rows narrowed to that value).
    A drop-down with a single possible value is shown greyed out.
    """
    options = sorted(rows[column].unique().tolist(), key=sort_key)

    source = default_rows if default_rows is not None and len(default_rows) else rows
    counts = source[column].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    preferred = next((v for v in counts.index if v in options), options[0])

    only_one = len(options) == 1
    value = st.selectbox(
        label.format(n=len(options)),
        options,
        index=options.index(preferred),
        format_func=format_func,
        key="|".join(path + [column]),
        disabled=only_one,
        help="Only one option exists for this selection in the dataset." if only_one else None,
    )
    if append_to_path:
        path.append(str(value))
    return value, rows[rows[column] == value]


def stretch_button(label):
    """Full-width primary button; works on old and new Streamlit versions."""
    try:
        return st.button(label, type="primary", width="stretch")
    except TypeError:                      # older Streamlit has no `width` argument
        return st.button(label, type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("💻 Laptop Price Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Laptop Price Prediction** notebook.
    Choose a laptop's specifications and the trained model will estimate its price in euros.

    Picking a model fills in its usual specifications, which you can then change. The lists
    avoid impossible laptops: the screen is tied to the model (size and resolution go together),
    and things like touchscreen, operating system and processor are limited to what the brand
    actually makes. A greyed-out list has only one possible value.
    """
)

if skipped_rows:
    st.warning(f"{skipped_rows} row(s) in laptop_price.csv could not be read and were left out of the lists.")
for message in model_messages:
    st.warning(f"While loading the model: {message}")
st.divider()

path = []   # the choices made so far (part of every widget key)

# ---- 1. Brand and model ---------------------------------------------------
st.subheader("1. Brand and model")
col1, col2 = st.columns(2)

with col1:
    company, company_rows = choose("Laptop Company", "Company", catalogue, path)
with col2:
    product, model_rows = choose("Model ({n} available)", "Product", company_rows, path)

# ---- 2. Type and screen: fixed by the model ------------------------------
st.subheader("2. Type and screen")
col1, col2 = st.columns(2)

rows = model_rows
with col1:
    type_name, rows = choose("Laptop Type", "TypeName", rows, path)
with col2:
    inches, rows = choose("Screen Size", "Inches", rows, path, format_func=lambda v: f'{v:g}"')
with col1:
    resolution, rows = choose("Resolution", "Resolution", rows, path,
                              format_func=resolution_label, sort_key=resolution_size)

typical_rows = rows            # this model with this screen: used for the starting values below
body_path = list(path)         # the lists below restart only when the model or its screen changes

# ---- 3. Screen features, OS, processor and graphics: limited by the brand -
# These lists are independent of each other, so changing one does not reset the others
# (except the CPU lists, which depend on each other: company -> family -> speed).
st.subheader("3. Features, processor and graphics")
col1, col2 = st.columns(2)

with col1:
    touch, _ = choose("Touchscreen", "Touchscreen", company_rows, body_path,
                      format_func=lambda v: "Yes" if v == 1 else "No",
                      default_rows=typical_rows, append_to_path=False)
with col2:
    ips, _ = choose("Screen Type", "IPS", company_rows, body_path,
                    format_func=lambda v: "IPS panel" if v == 1 else "Standard panel",
                    default_rows=typical_rows, append_to_path=False)
with col1:
    op_sys, _ = choose("Operating System", "OpSys", company_rows, body_path,
                       default_rows=typical_rows, append_to_path=False)
with col2:
    gpu_brand, _ = choose("Graphics Card Company", "Gpu_brand", company_rows, body_path,
                          default_rows=typical_rows, append_to_path=False)

cpu_path = list(body_path)
with col1:
    cpu_brand, cpu_rows = choose("CPU Company", "Cpu_brand", company_rows, cpu_path,
                                 default_rows=typical_rows)
with col2:
    cpu_fam, cpu_rows = choose("CPU Family", "Cpu_family", cpu_rows, cpu_path,
                               default_rows=typical_rows)
with col1:
    cpu_ghz, _ = choose("CPU Speed (GHz)", "Cpu_GHz", cpu_rows, cpu_path,
                        format_func=lambda v: f"{v:g}", default_rows=typical_rows)

# ---- 4. Memory and storage: limited by the laptop type -------------------
# Any amount seen in laptops of the same type (from any brand), so there is plenty to choose from.
st.subheader("4. Memory and storage")
type_rows = catalogue[catalogue["TypeName"] == type_name]
col1, col2 = st.columns(2)

with col1:
    ram, _ = choose("RAM", "Ram_GB", type_rows, body_path, format_func=lambda v: f"{v} GB",
                    default_rows=typical_rows, append_to_path=False)
with col2:
    ssd, _ = choose("SSD Storage", "SSD_GB", type_rows, body_path, format_func=storage_label,
                    default_rows=typical_rows, append_to_path=False)
with col1:
    hdd, _ = choose("Hard Disk (HDD) Storage", "HDD_GB", type_rows, body_path,
                    format_func=storage_label, default_rows=typical_rows, append_to_path=False)
with col2:
    flash, _ = choose("Flash Storage", "Flash_GB", type_rows, body_path,
                      format_func=storage_label, default_rows=typical_rows, append_to_path=False)
with col1:
    hybrid, _ = choose("Hybrid (SSHD) Storage", "Hybrid_GB", type_rows, body_path,
                       format_func=storage_label, default_rows=typical_rows, append_to_path=False)

st.divider()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if stretch_button("Predict Price"):
    res_w, res_h = (int(v) for v in resolution.split("x"))

    user_input = pd.DataFrame({
        "Inches": [float(inches)],
        "Touchscreen": [int(touch)],
        "Ram_GB": [int(ram)],
        "Cpu_GHz": [float(cpu_ghz)],
        "IPS": [int(ips)],
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

    # How many laptops in the dataset have exactly this configuration?
    same = pd.Series(True, index=catalogue.index)
    for column in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        if column != "PPI":
            same &= catalogue[column] == user_input.at[0, column]
    n_same = int(same.sum())

    st.success(f"### Estimated Price: €{predicted_price:,.0f}")
    if TYPICAL_ERROR_EUROS is not None:
        st.caption(f"Typical error of this model (cross-validation RMSE): about €{TYPICAL_ERROR_EUROS:,.0f}")
    if n_same:
        st.caption(f"This exact configuration appears {n_same} time(s) in the dataset the model learned from.")
    else:
        st.caption("This exact configuration is not in the dataset, so treat the estimate as approximate.")

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(user_input.T.rename(columns={0: "Value"}).astype(str))

st.divider()
st.caption(
    "Built for teaching purposes from a laptop price dataset (prices in euros). "
    "The estimate reflects the data the model was trained on, not today's market prices."
)

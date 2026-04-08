import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from typing import Optional
import random 

from ollama import chat
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Annotation",
    layout="wide",
    initial_sidebar_state="expanded",
)
# :root {
#     --bg-deep:      #ffffff;
#     --bg-card:      #13161e;
#     --bg-elevated:  #f0f2f6;
#     --border:       #252b3b;
#     --border-lit:   #3a4258;
#     --amber:        #f5a623;
#     --amber-dim:    #c47d10;
#     --green:        #4ade80;
#     --red:          #f87171;
#     --blue:         #60a5fa;
#     --text-primary: #1a1e2a;
#     --text-muted:   #7a8099;
#     --text-dim:     #4a5068;
#     --mono:         'Roboto', monospace;
#     --sans:         'IBM Plex Sans', sans-serif;
# }
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* Root theme */

:root {
    --bg-deep:      #ffffff;
    --bg-card:      #13161e;
    --bg-elevated:  #f0f2f6;
    --border:       #252b3b;
    --border-lit:   #c8cfe0;
    --text-primary: #1a1e2a;
    --text-muted:   #5a6080;
    --text-dim:     #9aa0b8;
    /* keep amber, green, red, blue as-is */
}
[theme]
base = "light"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#1a1e2a"

/* Global resets */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background-color: var(--bg-deep) !important;
}
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f0f2f6 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}
            

/* Header */
.lf-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}
.lf-logo {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--amber) !important;
    letter-spacing: -0.5px;
}
.lf-tagline {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Step pill */
.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-lit);
    border-radius: 4px;
    padding: 4px 12px;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}
.step-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--amber);
    display: inline-block;
}

/* Cards */
.lf-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.lf-card-title {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--amber);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}

/* Sample card */
.sample-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-lit);
    border-left: 3px solid var(--amber);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.6;
}
.sample-index {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Label badge */
.label-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 500;
    background: rgba(245,166,35,0.15);
    color: var(--amber);
    border: 1px solid rgba(245,166,35,0.3);
}
.label-badge.confirmed {
    background: rgba(74,222,128,0.12);
    color: var(--green);
    border-color: rgba(74,222,128,0.3);
}
.label-badge.uncertain {
    background: rgba(96,165,250,0.12);
    color: var(--blue);
    border-color: rgba(96,165,250,0.3);
}

/* Stat row */
.stat-row {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
}
.stat-box {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.75rem 1.25rem;
    flex: 1;
    min-width: 120px;
}
.stat-value {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--amber);
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-family: var(--mono);
}

/* Progress bar */
.progress-wrap {
    background: var(--bg-elevated);
    border-radius: 2px;
    height: 4px;
    margin: 0.5rem 0;
}
.progress-fill {
    height: 4px;
    border-radius: 2px;
    background: var(--amber);
    transition: width 0.4s ease;
}

/* Buttons */
.stButton > button {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-lit) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
    background: rgba(245,166,35,0.08) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: var(--amber) !important;
    color: #252b3b !important;
    border-color: var(--amber) !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--amber-dim) !important;
    color: #0d0f14 !important;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-lit) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
    font-family: var(--sans) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(245,166,35,0.12) !important;
}

/* Select box */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-lit) !important;
    color: var(--text-primary) !important;
}

/* Radio */
.stRadio label {
    font-family: var(--sans) !important;
    color: var(--text-primary) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* Alerts */
.stAlert {
    border-radius: 4px !important;
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
}

/* Tags container */
.tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0.5rem 0;
}
.category-tag {
    background: var(--bg-elevated);
    border: 1px solid var(--border-lit);
    color: var(--text-primary);
    padding: 4px 12px;
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
}
.category-tag:hover, .category-tag.active {
    background: rgba(245,166,35,0.15);
    border-color: var(--amber);
    color: var(--amber);
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.status-dot.online { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-dot.offline { background: var(--red); }

/* Mono text helpers */
.mono { font-family: var(--mono) !important; }
.muted { color: var(--text-muted) !important; }
.amber { color: var(--amber) !important; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Ollama helpers ─────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"

def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except:
        return False

def list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        gemma = [m for m in models if "gemma" in m.lower()]
        other = [m for m in models if "gemma" not in m.lower()]
        return gemma + other
    except:
        return []

def ollama_generate(model: str, prompt: str, system: str = "") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 1,
        "options": {"temperature": 0.1, "num_predict": 256}
    }
    if system:
        payload["system"] = system

    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e} | Response: {getattr(r, 'text', None)}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    prob = np.exp(r.json()["logprobs"][0]['logprob'])
    out = r.json().get("response", "").strip()
    return out, prob

def classify_text(model: str, text: str, categories: list[str],
                  few_shots: list[dict], task_description: str = "") -> dict:
    cats_str = ", ".join(f'"{c}"' for c in categories)
    
    few_shot_block = ""
    if few_shots:
        examples = []
        for s in few_shots:
            if s.get("label"):
                examples.append(f'Text: """{s["text"]}"""\nLabel: {s["label"]}')
        if examples:
            few_shot_block = "Examples:\n" + "\n\n".join(examples) + "\n\n"

    task_hint = f"Task context: {task_description}\n\n" if task_description else ""

    prompt = f"""{task_hint}{few_shot_block}Now classify the following text.

    Text: \"\"\"{text}\"\"\"

    Available categories: [{cats_str}]

    Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
    {{"label": "<one of the categories>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""

    raw,prob = ollama_generate(model, prompt)
    
    # Strip possible markdown fences
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        parsed = json.loads(raw)
        parsed["confidence"] = prob
        if parsed.get("label") not in categories:
            parsed["label"] = categories[0]
        return parsed
    except:
        return {"label": categories[0], "confidence": 0.0, "reasoning": "Parse error"}

def suggest_categories(model: str, texts: list[str], n: int = 5,
                        task_description: str = "") -> list[str]:
    MAX_CONTEXT = 128000
    text_len = sum(len(x) for x in texts)
    if text_len > MAX_CONTEXT:
        print(f"TEXT LENGTH TOO LONG WITH {text_len}")
    sample_size =  min(len(texts),1000)
    sample = random.sample(texts, sample_size)
    sample_str = "\n".join(f"- {t[:200]}" for t in sample)
    task_hint = f"Task context: {task_description}\n\n" if task_description else ""
    prompt = f"""{task_hint}Analyze these text samples and suggest {n} distinct, mutually-exclusive classification categories.

    Samples:
    {sample_str}

    Respond with ONLY a JSON array of category name strings. No markdown, no explanation.
    Example format: ["Category A", "Category B", "Category C"]"""
    
    raw, _ = ollama_generate(model, prompt)
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        cats = json.loads(raw)
        return [str(c).strip() for c in cats if c][:n]
    except:
        return ["Category 1", "Category 2", "Category 3"]

# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 1,
        "df": None,
        "text_column": None,
        "categories": [],
        "few_shots": [],  # list of {"text": ..., "label": ...}
        "results": [],
        "model": None,
        "task_description": "",
        "annotating_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Roboto',monospace; font-size:1.2rem; font-weight:600; 
                color:#f5a623; letter-spacing:-0.3px; margin-bottom:0.25rem;">
        Local Annotation
    </div>
    <div style="font-family:'Roboto',monospace; font-size:0.65rem; 
                color:#4a5068; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1.5rem;">
        LLM-Powered Annotation
    </div>
    """, unsafe_allow_html=True)

    # Ollama status
    online = ollama_available()
    status_color = "#4ade80" if online else "#f87171"
    status_text = "Ollama online" if online else "Ollama offline"
    st.markdown(f"""
    <div style="background:#f0f2f6; border:1px solid #ffffff; border-radius:4px; 
                padding:8px 12px; margin-bottom:1rem; font-family:'Roboto',monospace; 
                font-size:0.72rem; color:#7a8099;">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                     background:{status_color};margin-right:6px;vertical-align:middle;
                     {'box-shadow:0 0 6px '+status_color if online else ''}"></span>
        {status_text}
    </div>
    """, unsafe_allow_html=True)

    if online:
        models = list_models()
        if models:
            selected_model = st.selectbox("Model", models, 
                                           index=0,
                                           help="Gemma models shown first")
            st.session_state.model = selected_model
        else:
            st.warning("No models found. Pull one first:\n`ollama pull gemma3:4b`")
            st.session_state.model = None
    else:
        st.error("Start Ollama: `ollama serve`")
        st.session_state.model = None
        st.markdown("""
        <div style="font-size:0.75rem; color:#7a8099; margin-top:0.5rem; 
                    font-family:'Roboto',monospace;">
        Recommended models:<br>
        • gemma3:4b<br>
        • gemma3:12b<br>
        • gemma3:27b
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Step navigator
    st.markdown("""
    <div style="font-family:'Roboto',monospace; font-size:0.65rem; 
                color:#4a5068; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;">
        Workflow
    </div>
    """, unsafe_allow_html=True)
    
    steps = [
        (1, "Upload Data"),
        (2, "Define Categories"),
        (3, "Annotate Samples"),
        (4, "Classify"),
        (5, "Review & Export"),
    ]
    
    for num, label in steps:
        is_current = st.session_state.step == num
        color = "#f5a623" if is_current else "#4a5068"
        weight = "600" if is_current else "400"
        prefix = "▶ " if is_current else f"{num}. "
        st.markdown(f"""
        <div style="font-family:'Roboto',monospace; font-size:0.75rem; 
                    color:{color}; font-weight:{weight}; padding:3px 0; 
                    letter-spacing:0.03em;">
            {prefix}{label}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Task description
    st.markdown("""<div style="font-family:'Roboto',monospace; font-size:0.65rem; 
                color:#4a5068; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem;">
    Task Description</div>""", unsafe_allow_html=True)
    task_desc = st.text_area("", value=st.session_state.task_description,
                              placeholder="Optional: describe your classification task to guide the model...",
                              height=90, label_visibility="collapsed")
    st.session_state.task_description = task_desc

    if st.button("↺ Reset Everything", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="lf-header">
    <span class="lf-logo">Local Annotation</span>
    <span class="lf-tagline">Few-shot classification · Ollama · Local LLM</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload Data
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown('<div class="step-pill"><span class="step-dot"></span>Step 1 — Upload Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown('<div class="lf-card-title">Upload your dataset</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "CSV or Excel file", 
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            help="Upload a CSV or Excel file containing a column with text to classify"
        )
        
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                st.session_state.df = df
                st.success(f"✓ Loaded **{len(df):,}** rows · **{len(df.columns)}** columns")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown("---")
            st.markdown('<div class="lf-card-title">Select text column</div>', unsafe_allow_html=True)
            
            text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
            if not text_cols:
                text_cols = df.columns.tolist()
            
            col_pick = st.selectbox("Column containing text to classify:", text_cols, 
                                     label_visibility="collapsed")
            st.session_state.text_column = col_pick
            
            # Preview
            st.markdown("---")
            st.markdown('<div class="lf-card-title">Preview</div>', unsafe_allow_html=True)
            preview = df[[col_pick]].head(5).copy()
            preview.columns = ["Text"]
            preview["Text"] = preview["Text"].astype(str).str[:120] + "..."
            st.dataframe(preview, use_container_width=True, hide_index=True)
    
    with col2:
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown("""
            <div style="background:#f0f2f6; border:1px solid #ffffff; border-radius:6px; padding:1.25rem 1.5rem;">
            """, unsafe_allow_html=True)
            st.markdown('<div class="lf-card-title">Dataset summary</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-value">{len(df):,}</div>
                    <div class="stat-label">Rows</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(df.columns)}</div>
                    <div class="stat-label">Columns</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.text_column:
                col_data = df[st.session_state.text_column].astype(str)
                avg_len = int(col_data.str.len().mean())
                null_count = df[st.session_state.text_column].isna().sum()
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-box">
                        <div class="stat-value">{avg_len}</div>
                        <div class="stat-label">Avg chars</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{null_count}</div>
                        <div class="stat-label">Nulls</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#f0f2f6; border:1px dashed #ffffff; border-radius:6px; 
                        padding:2rem; text-align:center; color:#4a5068;
                        font-family:'Roboto',monospace; font-size:0.8rem; line-height:2;">
                ↑ Upload a file to get started<br><br>
                Supported formats:<br>
                .csv · .xlsx · .xls
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Continue →", type="primary", disabled=(st.session_state.df is None or st.session_state.text_column is None)):
            st.session_state.step = 2
            st.rerun()
    st.markdown("---")
    st.markdown('<div class="lf-card-title">Preprocessing</div>', unsafe_allow_html=True)

    remove_nulls = st.checkbox("Remove rows with empty text", value=True)
    # strip_whitespace = st.checkbox("Strip leading/trailing whitespace", value=True)
    # drop_duplicates = st.checkbox("Drop duplicate texts", value=False)
    min_chars = st.slider("Minimum character length (0 = no filter)", 0, 500, 0)
    if st.button("Apply preprocessing"):
        if not uploaded:
            st.write("Please upload file first")
        else: 
            original_len = len(st.session_state.df)
            working = st.session_state.df.copy()
            col = st.session_state.text_column

            # Step 1: remove real NaN/None FIRST (before any str casting)
            if remove_nulls:
                working = working[pd.notna(working[col])]

            # Step 2: strip whitespace
            # if strip_whitespace:
            #     working[col] = working[col].astype(str).str.strip()

            # Step 3: second pass — catch empty strings + string variants of null
            if remove_nulls:
                null_strings = {"nan", "none", "null", "n/a", "na", "-", ""}
                working = working[~working[col].str.strip().str.lower().isin(null_strings)]

            # Step 4: min character filter
            if min_chars > 0:
                working = working[working[col].astype(str).str.len() >= min_chars]

            # Step 5: dedup
            # if drop_duplicates:
            #     working = working.drop_duplicates(subset=[col])

            working = working.reset_index(drop=True)
            st.session_state.df = working
            removed = original_len - len(working)
            st.success(f"✓ Removed {removed} rows → {len(working):,} remaining")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Define Categories
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown('<div class="step-pill"><span class="step-dot"></span>Step 2 — Define Categories</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown('<div class="lf-card-title">How do you want to define categories?</div>', unsafe_allow_html=True)
        mode = st.radio("", ["✏️  Define manually", "🤖  Auto-suggest with AI"], 
                         label_visibility="collapsed")
        
        st.markdown("---")
        
        if mode == "✏️  Define manually":
            st.markdown('<div class="lf-card-title">Enter categories</div>', unsafe_allow_html=True)
            cats_input = st.text_area(
                "",
                value="\n".join(st.session_state.categories),
                placeholder="One category per line, e.g.:\nPositive\nNegative\nNeutral",
                height=200,
                label_visibility="collapsed"
            )
            if st.button("Save categories", type="primary"):
                cats = [c.strip() for c in cats_input.strip().split("\n") if c.strip()]
                if len(cats) >= 2:
                    st.session_state.categories = cats
                    st.success(f"✓ Saved {len(cats)} categories")
                else:
                    st.warning("Add at least 2 categories.")
        
        else:  # Auto-suggest
            if not st.session_state.model:
                st.warning("Select a model in the sidebar first.")
            else:
                n_cats = st.slider("Number of categories to suggest", 2, 10, 5)
                if st.button("🤖 Generate categories", type="primary"):
                    texts = st.session_state.df[st.session_state.text_column].dropna().astype(str).tolist()
                    with st.spinner("Analyzing your data..."):
                        suggested = suggest_categories(
                            st.session_state.model, texts, n=n_cats,
                            task_description=st.session_state.task_description
                        )
                    st.session_state.categories = suggested
                    st.success(f"✓ Generated {len(suggested)} categories")
                
                if st.session_state.categories:
                    st.markdown("---")
                    st.markdown('<div class="lf-card-title">Edit suggested categories</div>', unsafe_allow_html=True)
                    cats_edit = st.text_area(
                        "",
                        value="\n".join(st.session_state.categories),
                        height=180,
                        label_visibility="collapsed"
                    )
                    if st.button("Update", use_container_width=True):
                        cats = [c.strip() for c in cats_edit.strip().split("\n") if c.strip()]
                        st.session_state.categories = cats
                        st.success("✓ Updated")
    
    with col2:
        st.markdown('<div class="lf-card-title">Current categories</div>', unsafe_allow_html=True)
        if st.session_state.categories:
            colors = ["#f5a623", "#60a5fa", "#4ade80", "#f472b6", "#a78bfa",
                      "#fb923c", "#34d399", "#38bdf8", "#e879f9", "#facc15"]
            for i, cat in enumerate(st.session_state.categories):
                c = colors[i % len(colors)]
                st.markdown(f"""
                <div style="background:#f0f2f6; border:1px solid #ffffff; 
                            border-left:3px solid {c}; border-radius:4px;
                            padding:8px 14px; margin-bottom:6px;
                            font-family:'Roboto',monospace; font-size:0.82rem;
                            color:#1a1e2a;">
                    {cat}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#4a5068; font-family:'Roboto',monospace; 
                        font-size:0.78rem; padding:1rem 0;">
                No categories defined yet.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    col_back, col_next, _ = st.columns([1, 1, 3])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Continue →", type="primary", 
                      disabled=(len(st.session_state.categories) < 2)):
            st.session_state.step = 3
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Annotate Few-Shot Samples
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    st.markdown('<div class="step-pill"><span class="step-dot"></span>Step 3 — Annotate Few-Shot Samples</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    text_col = st.session_state.text_column
    categories = st.session_state.categories
    few_shots = st.session_state.few_shots

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        <div style="color:#7a8099; font-size:0.88rem; margin-bottom:1rem; line-height:1.6;">
            Label a handful of examples by hand. These will be used as few-shot examples 
            to guide the model during classification. More examples = better accuracy.
        </div>
        """, unsafe_allow_html=True)
        
        # Pick how many samples to annotate
        n_samples = st.slider("How many samples to annotate?", 1, min(20, len(df)), 
                               min(5, len(df)))
        
        # Sample selection strategy
        strategy = st.radio("Sample selection", 
                             ["First N rows", "Random sample"],
                             horizontal=True)
        
        if strategy == "Random sample":
            seed = st.number_input("Random seed", value=42, step=1)
            samples = df[text_col].dropna().sample(n=n_samples, random_state=int(seed)).tolist()
        else:
            samples = df[text_col].dropna().head(n_samples).tolist()
        
        st.markdown("---")
        st.markdown('<div class="lf-card-title">Label each sample</div>', unsafe_allow_html=True)
        
        # Build annotation UI
        new_shots = []
        for i, text in enumerate(samples):
            text_str = str(text)
            existing = next((s for s in few_shots if s["text"] == text_str), None)
            default_label = existing["label"] if existing else None
            
            st.markdown(f"""
            <div class="sample-card">
                <div class="sample-index">Sample {i+1} of {n_samples}</div>
                {text_str[:400]}{'...' if len(text_str) > 400 else ''}
            </div>
            """, unsafe_allow_html=True)
            
            options = ["(skip)"] + categories
            default_idx = options.index(default_label) if default_label in options else 0
            
            label = st.selectbox(
                f"Label for sample {i+1}",
                options=options,
                index=default_idx,
                key=f"shot_{i}",
                label_visibility="collapsed"
            )
            
            if label != "(skip)":
                new_shots.append({"text": text_str, "label": label})
            
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        
        if st.button("Save annotations", type="primary", use_container_width=True):
            st.session_state.few_shots = new_shots
            st.success(f"✓ Saved {len(new_shots)} annotations")

    with col2:
        st.markdown('<div class="lf-card-title">Annotation summary</div>', unsafe_allow_html=True)
        
        current_shots = st.session_state.few_shots
        
        st.markdown(f"""
        <div class="stat-box" style="margin-bottom:0.75rem;">
            <div class="stat-value">{len(current_shots)}</div>
            <div class="stat-label">Labeled samples</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show distribution
        if current_shots:
            from collections import Counter
            dist = Counter(s["label"] for s in current_shots)
            colors = ["#f5a623", "#60a5fa", "#4ade80", "#f472b6", "#a78bfa",
                      "#fb923c", "#34d399", "#38bdf8", "#e879f9", "#facc15"]
            
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="lf-card-title">Label distribution</div>', unsafe_allow_html=True)
            
            for i, (label, count) in enumerate(dist.most_common()):
                c = colors[i % len(colors)]
                pct = count / len(current_shots) * 100
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; 
                                font-family:'Roboto',monospace; font-size:0.72rem;
                                color:#7a8099; margin-bottom:3px;">
                        <span style="color:#1a1e2a;">{label}</span>
                        <span>{count}</span>
                    </div>
                    <div style="background:#f0f2f6; border-radius:2px; height:4px;">
                        <div style="width:{pct:.0f}%; height:4px; border-radius:2px; background:{c};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.78rem; color:#7a8099; line-height:1.7; 
                    font-family:'Roboto',monospace;">
            💡 Tips:<br>
            • 3-5 examples per category works well<br>
            • Try to cover edge cases<br>
            • You can skip and classify with 0 examples (zero-shot)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    col_back, col_next, _ = st.columns([1, 1, 3])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 2
            st.rerun()
    with col_next:
        if st.button("Continue →", type="primary"):
            st.session_state.step = 4
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Classify
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    st.markdown('<div class="step-pill"><span class="step-dot"></span>Step 4 — Classify</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    text_col = st.session_state.text_column
    categories = st.session_state.categories
    few_shots = st.session_state.few_shots
    model = st.session_state.model

    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.markdown('<div class="lf-card-title">Classification settings</div>', unsafe_allow_html=True)
        
        # Config
        st.markdown(f"""
        <div style="background:#f0f2f6; border:1px solid #ffffff; border-radius:4px; 
                    padding:1rem; margin-bottom:1rem; font-family:'Roboto',monospace; 
                    font-size:0.78rem; line-height:2; color:#7a8099;">
            Model: <span style="color:#f5a623">{model or 'None selected'}</span><br>
            Categories: <span style="color:#1a1e2a">{len(categories)}</span><br>
            Few-shot examples: <span style="color:#1a1e2a">{len(few_shots)}</span><br>
            Total rows: <span style="color:#1a1e2a">{len(df):,}</span>
        </div>
        """, unsafe_allow_html=True)
        
        max_rows = st.number_input(
            "Max rows to classify (0 = all)", 
            min_value=0, max_value=len(df), value=min(50, len(df)), step=10
        )
        
        skip_existing = st.checkbox("Skip already classified rows", value=True)
        
        if not model:
            st.error("No model selected. Check sidebar.")
        elif len(categories) < 2:
            st.warning("Define at least 2 categories first.")
        else:
            if st.button("🚀 Start classification", type="primary", use_container_width=True):
                null_strings = {"nan", "none", "null", "n/a", "na", "-", ""}
                mask = pd.notna(df[text_col]) & \
                    (~df[text_col].astype(str).str.strip().str.lower().isin(null_strings))

                all_texts = [(i, str(t)) for i, t in enumerate(df[text_col]) if mask.iloc[i]]
                if max_rows > 0:
                    all_texts = all_texts[:max_rows]
                
                results = list(st.session_state.results) if st.session_state.results else []
                existing_indices = {r["index"] for r in results} if skip_existing else set()
                
                to_classify = [(i, t) for i, t in all_texts 
                    if i not in existing_indices]
                
                if not to_classify:
                    st.info("All selected rows already classified.")
                else:
                    progress_bar = st.progress(0)
                    status_text_el = st.empty()
                    
                    for done, (idx, text) in enumerate(to_classify):
                        status_text_el.markdown(
                            f'<div style="font-family:\'Roboto\',monospace;font-size:0.75rem;'
                            f'color:#7a8099;">Classifying row {idx+1} / {len(all_texts)}...</div>',
                            unsafe_allow_html=True
                        )
                        try:
                            result = classify_text(
                                model, text, categories, few_shots,
                                task_description=st.session_state.task_description
                            )
                            result["index"] = idx
                            result["text"] = text[:300]
                            results.append(result)
                        except Exception as e:
                            results.append({
                                "index": idx,
                                "text": text[:300],
                                "label": "Error",
                                "confidence": 0.0,
                                "reasoning": str(e)
                            })
                        
                        progress_bar.progress((done + 1) / len(to_classify))
                    
                    st.session_state.results = results
                    status_text_el.markdown(
                        f'<div style="color:#4ade80;font-family:\'Roboto\',monospace;'
                        f'font-size:0.78rem;">✓ Classified {len(to_classify)} rows</div>',
                        unsafe_allow_html=True
                    )
                    # time.sleep(0.5) <-- not necessary when ran locally
                    st.session_state.step = 5
                    st.rerun()

    with col2:
        st.markdown('<div class="lf-card-title">Few-shot examples in use</div>', unsafe_allow_html=True)
        if few_shots:
            for shot in few_shots[:6]:
                st.markdown(f"""
                <div style="background:#f0f2f6; border:1px solid #ffffff; border-left:3px solid #f5a623;
                            border-radius:4px; padding:8px 14px; margin-bottom:6px; font-size:0.82rem; line-height:1.5;">
                    <div style="font-family:'Roboto',monospace; font-size:0.65rem; 
                                color:#4a5068; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;">
                        → {shot['label']}
                    </div>
                    {str(shot['text'])[:150]}...
                </div>
                """, unsafe_allow_html=True)
            if len(few_shots) > 6:
                st.markdown(f'<div style="color:#4a5068;font-family:\'Roboto\',monospace;font-size:0.72rem;">+{len(few_shots)-6} more</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#7a8099; font-family:'Roboto',monospace; font-size:0.78rem;
                        padding:1rem; background:#f0f2f6; border-radius:4px; line-height:1.8;">
                No few-shot examples.<br>
                Running in <span style="color:#f5a623">zero-shot</span> mode.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 3
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Review & Export
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    st.markdown('<div class="step-pill"><span class="step-dot"></span>Step 5 — Review & Export</div>', unsafe_allow_html=True)
    
    results = st.session_state.results
    df = st.session_state.df
    
    if not results:
        st.warning("No results yet. Go to Step 4 to run classification.")
    else:
        results_df = pd.DataFrame(results)
        
        # Stats
        from collections import Counter
        label_counts = Counter(r["label"] for r in results)
        avg_conf = sum(r.get("confidence", 0) for r in results) / len(results)
        
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-value">{len(results):,}</div>
                <div class="stat-label">Classified</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{avg_conf:.0%}</div>
                <div class="stat-label">Avg confidence</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(label_counts)}</div>
                <div class="stat-label">Labels used</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Label distribution
        col1, col2 = st.columns([2, 3], gap="large")
        
        with col1:
            st.markdown('<div class="lf-card-title">Label distribution</div>', unsafe_allow_html=True)
            colors = ["#f5a623", "#60a5fa", "#4ade80", "#f472b6", "#a78bfa",
                      "#fb923c", "#34d399", "#38bdf8", "#e879f9", "#facc15"]
            
            for i, (label, count) in enumerate(label_counts.most_common()):
                c = colors[i % len(colors)]
                pct = count / len(results) * 100
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; 
                                font-family:'Roboto',monospace; font-size:0.78rem;
                                margin-bottom:4px;">
                        <span style="color:#1a1e2a;">{label}</span>
                        <span style="color:{c};">{count} ({pct:.0f}%)</span>
                    </div>
                    <div style="background:#f0f2f6; border-radius:2px; height:6px;">
                        <div style="width:{pct:.0f}%; height:6px; border-radius:2px; background:{c};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="lf-card-title">Filter & review results</div>', unsafe_allow_html=True)
            
            filter_label = st.selectbox("Filter by label", ["All"] + list(label_counts.keys()))
            conf_threshold = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
            
            filtered = [r for r in results 
                        if (filter_label == "All" or r["label"] == filter_label)
                        and r.get("confidence", 0) >= conf_threshold]
            
            st.markdown(f'<div style="font-family:\'Roboto\',monospace;font-size:0.72rem;color:#7a8099;margin-bottom:0.75rem;">{len(filtered)} results shown</div>', unsafe_allow_html=True)
            
            for r in filtered[:10]:
                conf = r.get("confidence", 0)
                conf_color = "#4ade80" if conf >= 0.8 else "#f5a623" if conf >= 0.5 else "#f87171"
                st.markdown(f"""
                <div style="background:#f0f2f6; border:1px solid #ffffff; border-radius:4px;
                            padding:10px 14px; margin-bottom:6px; font-size:0.82rem; line-height:1.5;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-family:'Roboto',monospace; font-size:0.72rem;
                                     background:rgba(245,166,35,0.12); color:#f5a623;
                                     border:1px solid rgba(245,166,35,0.25); border-radius:3px;
                                     padding:2px 8px;">{r['label']}</span>
                        <span style="font-family:'Roboto',monospace; font-size:0.72rem;
                                     color:{conf_color};">{conf:.0%}</span>
                    </div>
                    <div style="color:#1a1e2a; margin-bottom:4px;">{str(r.get('text',''))[:200]}</div>
                    <div style="color:#4a5068; font-size:0.75rem; font-style:italic;">{r.get('reasoning','')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if len(filtered) > 10:
                st.markdown(f'<div style="color:#4a5068;font-family:\'Roboto\',monospace;font-size:0.72rem;">+{len(filtered)-10} more (export to see all)</div>', unsafe_allow_html=True)
        
        # Export
        st.markdown("---")
        st.markdown('<div class="lf-card-title">Export results</div>', unsafe_allow_html=True)
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        # Merge with original df
        result_map = {r["index"]: r for r in results}
        export_df = df.copy().reset_index(drop=True)
        export_df["predicted_label"] = export_df.index.map(lambda i: result_map.get(i, {}).get("label", ""))
        export_df["confidence"] = export_df.index.map(lambda i: result_map.get(i, {}).get("confidence", ""))
        export_df["reasoning"] = export_df.index.map(lambda i: result_map.get(i, {}).get("reasoning", ""))
        
        with col_exp1:
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download CSV",
                data=csv_data,
                file_name="Local Annotation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with col_exp2:
            shots_json = json.dumps(st.session_state.few_shots, indent=2, ensure_ascii=False)
            st.download_button(
                "⬇ Export few-shot examples",
                data=shots_json.encode("utf-8"),
                file_name="few_shots.json",
                mime="application/json",
                use_container_width=True,
            )
        
        with col_exp3:
            config = {
                "model": st.session_state.model,
                "categories": st.session_state.categories,
                "task_description": st.session_state.task_description,
                "few_shots": st.session_state.few_shots,
            }
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            st.download_button(
                "⬇ Export config",
                data=config_json.encode("utf-8"),
                file_name="Local Annotation_config.json",
                mime="application/json",
                use_container_width=True,
            )
    
    st.markdown("---")
    col_back, col_more, _ = st.columns([1, 1, 3])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 4
            st.rerun()
    with col_more:
        if st.button("Classify more →", type="primary"):
            st.session_state.step = 4
            st.rerun()

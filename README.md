# 🔬 LabelForge

**LLM-powered few-shot text classification with Ollama — runs 100% locally.**

---

## Features

- 📁 Upload CSV or Excel files
- 🤖 Auto-suggest categories using your data + an LLM
- ✏️ Define custom categories manually
- 🏷️ Annotate few-shot examples by hand
- ⚡ Classify your dataset with Gemma (or any Ollama model)
- 📊 Review results with confidence scores & reasoning
- ⬇️ Export annotated CSV + config

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install & start Ollama

```bash
# Install Ollama from https://ollama.com
ollama serve
```

### 3. Pull a model (Gemma recommended)

```bash
# Fast (4B) — good for most tasks
ollama pull gemma3:4b

# Better accuracy
ollama pull gemma3:12b

# Best accuracy (needs ~16GB RAM)
ollama pull gemma3:27b
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Workflow

```
Upload CSV/XLSX
     ↓
Define categories (manual or AI-suggested)
     ↓
Annotate few-shot samples by hand
     ↓
Run classification
     ↓
Review & export
```

---

## Tips

- **Zero-shot**: Skip the annotation step entirely — works fine for simple tasks
- **Few-shot**: 3-5 examples per category significantly improves accuracy
- **Task description**: Add a task description in the sidebar to guide the model
- **Confidence threshold**: Filter low-confidence results for human review
- **Config export**: Save your categories + few-shots to reproduce results later

---

## Model recommendations

| Model | RAM | Speed | Quality |
|-------|-----|-------|---------|
| gemma3:4b | ~4GB | Fast | Good |
| gemma3:12b | ~10GB | Medium | Better |
| gemma3:27b | ~20GB | Slow | Best |

Any other Ollama model works too (llama3, mistral, phi3, etc.)

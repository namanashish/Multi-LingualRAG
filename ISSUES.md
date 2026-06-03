# ISSUES — Problems Encountered and Fixes

All dependency conflicts, errors, and fixes encountered during development of the MultiLingual RAG pipeline.

---

## Critical Dependency Warning

> ⚠️ This project has a **3-way dependency conflict** between `faster-whisper`, `IndicTrans2`, and `sentence-transformers`. Read Issue #5 and #6 carefully before installing anything.

---

## Issue 1 — CUDA Version Confusion

**Error:** No direct error but wrong PyTorch wheel installed.

**Cause:** `nvidia-smi` shows the maximum CUDA version your driver *supports*, not the CUDA toolkit version actually installed on the system. Driver 581.95 shows CUDA 13.0 which doesn't exist as a PyTorch wheel.

**Explanation:**
```
nvidia-smi CUDA 13.0  ← maximum supported by driver
PyTorch cu121         ← actual runtime bundled inside wheel
These are independent. cu121 wheel works on any driver ≥ 525.x
```

**Fix:** Always install PyTorch with `cu121` wheel regardless of what `nvidia-smi` shows.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify:**
```python
import torch
print(torch.cuda.is_available())       # must be True
print(torch.cuda.get_device_name(0))   # must show your GPU
```

---

## Issue 2 — Python Version Incompatibility

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torch
ERROR: No matching distribution found for faster-whisper
```

**Cause:** Python 3.14 is a pre-release build. PyTorch, faster-whisper, transformers, and most ML libraries do not publish wheels for 3.14.

**Fix:** Install Python 3.11 specifically. Use `py` launcher to target the right version.

```bash
py -0                           # list all installed Python versions
py -3.11 -m venv raprod         # create venv with Python 3.11
raprod\Scripts\activate
python --version                # verify: Python 3.11.x
```

---

## Issue 3 — Hindi Detected as Urdu

**Symptom:** Audio in Hindi transcribed correctly but `info.language` returns `"ur"` (Urdu) instead of `"hi"` (Hindi).

**Cause:** Hindi and Urdu are both dialects of Hindustani — identical spoken sounds, different scripts. Whisper's language detector cannot reliably distinguish them from audio alone.

**Fix:** Force the language explicitly in the transcribe call.

```python
segments, info = model.transcribe(
    audio_file,
    language="hi",          # force Hindi — do not leave as None
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)
```

---

## Issue 4 — IndicTrans2 Missing `transformers.onnx` Module

**Error:**
```
ModuleNotFoundError: No module named 'transformers.onnx'
File: configuration_indictrans.py line 23
    from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
```

**Cause:** HuggingFace removed the `transformers.onnx` submodule in transformers 5.x. IndicTrans2's model configuration file still imports from it.

**Fix:** Downgrade transformers to 4.40.0 which still has the `onnx` submodule.

```bash
pip install transformers==4.40.0 huggingface-hub==0.36.2 tokenizers==0.19.1
```

> ⚠️ This creates a conflict with sentence-transformers — see Issue 6.

---

## Issue 5 — IndicTransToolkit Wrong Import Path

**Error:**
```
ImportError: cannot import name 'IndicProcessor' from 'IndicTransToolkit'
```

**Cause:** Newer version of IndicTransToolkit moved `IndicProcessor` into a submodule.

**Fix:**

```python
# wrong
from IndicTransToolkit import IndicProcessor

# correct
from IndicTransToolkit.processor import IndicProcessor
```

---

## Issue 6 — Three-Way Dependency Conflict (Critical)

This is the most important issue in the project. Three packages require incompatible versions of `transformers`:

| Package | Requires transformers |
|---|---|
| faster-whisper 1.2.1 | uses ctranslate2, NOT transformers directly ✓ |
| IndicTrans2 | requires transformers < 5.0 (needs `transformers.onnx`) |
| sentence-transformers 5.x | requires transformers >= 4.41.0 |
| sentence-transformers 2.7.0 | requires transformers >= 4.34.0, < 5.0 ✓ |

**The conflict:**
```
IndicTrans2           needs transformers == 4.40.0
sentence-transformers 5.x needs transformers >= 4.41.0
→ impossible to satisfy both simultaneously
```

**Resolution:**
```
faster-whisper   → safe, uses ctranslate2 not transformers
IndicTrans2      → pin transformers == 4.40.0
sentence-transf  → downgrade to 2.7.0 (compatible with 4.40.0)
```

**Install order that works:**

```bash
# 1. Pin transformers first
pip install transformers==4.40.0 huggingface-hub==0.36.2 tokenizers==0.19.1

# 2. Install sentence-transformers at compatible version
pip install sentence-transformers==2.7.0

# 3. Never run plain `pip install sentence-transformers` — it pulls 5.x
```

**Warning — installs that silently break this:**

Any of these will upgrade transformers back to 5.x:
```bash
pip install sentence-transformers        # ← upgrades to 5.x
pip install langchain-community          # ← may pull latest transformers
pip install chromadb                     # ← may pull latest transformers
```

**After every major install batch, always verify and re-pin:**

```bash
pip list | findstr transformers
# if it shows 5.x, run:
pip install transformers==4.40.0 huggingface-hub==0.36.2 tokenizers==0.19.1
```

---

## Issue 7 — `parser.py` Circular Import with Python Built-in

**Error:**
```
AttributeError: partially initialized module 'torch' has no attribute 'version'
(most likely due to a circular import)
```

**Cause:** Created a file named `parser.py` in the project. Python's built-in `parser` module has the same name. When importing, Python partially initializes the wrong module causing a circular import chain that breaks torch.

**Fix Option A (recommended):** Rename the file to `doc_parser.py`.

**Fix Option B:** Keep `parser.py` but place it in a `src/` subdirectory and use explicit `sys.path` injection:

```python
import sys
sys.path.insert(0, "C:/path/to/your/project/src")
from parser import DocumentParser
```

> Note: `sys.path.insert(0, ...)` (position 0) ensures your `src/` folder is checked before Python's built-in paths.

---

## Issue 8 — LangChain Import Paths Changed

**Error:**
```
ModuleNotFoundError: No module named 'langchain.text_splitter'
ModuleNotFoundError: No module named 'langchain_community.embeddings'
```

**Cause:** LangChain restructured its package into separate sub-packages across versions. Old tutorials use the monolithic import paths which no longer exist.

**Fix:**

```python
# text splitter
# wrong
from langchain.text_splitter import RecursiveCharacterTextSplitter
# correct
from langchain_text_splitters import RecursiveCharacterTextSplitter

# embeddings
# wrong
from langchain_community.embeddings import HuggingFaceEmbeddings
# correct
from langchain_huggingface import HuggingFaceEmbeddings
```

```bash
pip install langchain-huggingface langchain-text-splitters
```

---

## Issue 9 — Parser Key Names Changed Breaking Downstream Code

**Error:**
```
KeyError: 'page_num'
KeyError: 'method'
```

**Cause:** During parser rewrite to add table/list/section extraction, key names in the returned chunk dictionaries were changed without updating downstream `rag.py`.

| Old key | New key | Meaning |
|---|---|---|
| `page_num` | `page` | page number |
| `method` | `type` | how content was extracted |

**Fix:** Update all references in `rag.py`:

```python
# old
all_metadata.append({
    "page":   page["page_num"],
    "method": page["method"],
})

# correct
all_metadata.append({
    "page": page["page"],
    "type": page["type"],
})
```

---

## Issue 10 — Tesseract Not in System PATH

**Symptom:**
```
TesseractNotFoundError: tesseract is not installed or not in PATH
```

**Cause:** Tesseract is installed but not added to Windows PATH.

**Fix:** Hardcode the path in code instead of relying on PATH.

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Verify installation path:
```bash
dir "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## Quick Diagnostic Checklist

If something breaks, run through this in order:

```bash
# 1. Check Python version in venv
python --version          # must be 3.11.x

# 2. Check transformers version
pip show transformers     # must be 4.40.0

# 3. Check sentence-transformers version
pip show sentence-transformers   # must be 2.7.0

# 4. Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 5. Check Tesseract
tesseract --version       # or check dir directly

# 6. Check ffmpeg
ffmpeg -version
```


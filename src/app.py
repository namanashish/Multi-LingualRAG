import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

print("Loading Whisper model...")
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
print("Whisper loaded!")

audio_file = "../audio/test3.m4a"

print(f"\nTranscribing {audio_file}...")
segments, info = whisper_model.transcribe(
    audio_file,
    beam_size=5,
    language="hi",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)

print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
print("\n--- Transcription ---")

full_text = ""
for segment in segments:
    print(f"[{segment.start:.1f}s → {segment.end:.1f}s] {segment.text}")
    full_text += segment.text + " "

full_text = full_text.strip()
print(f"\nFull text:\n{full_text}")

print("\nLoading IndicTrans2 model (CPU)... this may take a minute first time")

model_name = "ai4bharat/indictrans2-indic-en-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32 
).to("cpu")

ip = IndicProcessor(inference=True)

print("Translating Hindi → English...")

batch = ip.preprocess_batch(
    [full_text],
    src_lang="hin_Deva",   
    tgt_lang="eng_Latn"    
)

inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True
).to("cpu")

with torch.no_grad():
    generated_tokens = trans_model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=256,
        num_beams=5,
        num_return_sequences=1
    )

with tokenizer.as_target_tokenizer():
    generated_tokens = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

translations = ip.postprocess_batch(generated_tokens, lang="eng_Latn")

print("\n--- Translation (English) ---")
print(translations[0])

import sys 
sys.path.append("C:/Users/naman/Downloads/Projects/MulitLingualRAG/src")
from cleaner import TextCleaner
cleaner = TextCleaner()
raw = translations[0]
clean= cleaner.clean(raw)
print("Raw : " ,raw)
print("Clean : ",clean)

import sys
sys.path.append("C:/Users/naman/Downloads/Projects/MulitLingualRAG/src")
from router import IntentRouter 
router = IntentRouter()
queries = [
    "What documents do I need for Kisan Credit Card?",
    "I want to apply for a new ration card",
    "What are the rules under PM Kisan scheme?",
    "Register me for PM Kisan Yojana"
]
for q in queries :
    result = router.route(q)
    print(result)

import sys
sys.path.append("C:/Users/naman/Downloads/Projects/MulitLingualRAG")

from parser import DocumentParser

parser = DocumentParser(poppler_path=r"C:\poppler\Library\bin")

pages = parser.parse("../docs/NA_CV.pdf")

for chunk in pages:
    print(f"Page {chunk['page']} ({chunk['type']}):")
    print(chunk['text'][:300])
    print("---")

print("------")
import sys
sys.path.append("C:/Users/naman/Downloads/Projects/MulitLingualRAG/src")
print("------")
from rag import RAGPipeline
print("-----")
rag = RAGPipeline(docs_folder="../docs")

# First time — ingest your PDFs
rag.ingest()

# Test retrieval
results = rag.retrieve("Multi Factor authentication")

for r in results:
    print(f"Score: {r['score']} | Source: {r['source']} | Page: {r['page']}")
    print(r['text'][:300])
    print("---")

import sys
sys.path.append("C:/Users/naman/Downloads/Projects/MulitLingualRAG/src")

from parser import DocumentParser

parser = DocumentParser(poppler_path=r"C:\poppler\Library\bin")
chunks = parser.parse("../docs/NA_CV.pdf")

print(f"Total chunks: {len(chunks)}")
for chunk in chunks[:5]:
    print(f"Page {chunk['page']} | Type: {chunk['type']} | Section: {chunk['section']}")
    print(chunk['text'][:300])
    print("---")

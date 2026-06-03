from transformers import pipeline
class TextCleaner:
    def __init__(self):
        self.cleaner = pipeline ("text2text-generation", model="google/flan-t5-large",device="cpu")
    def clean(self, text:str)->str:
        prompt=f"Translate any remaining Hindi or Indic language word into English in this sentence, keep the meaning intact: {text}"
        result = self.cleaner(prompt,
                             max_length = 256,
                             num_beams = 4,
                             repetition_penalty=1.2)[0]["generated_text"],
        return result

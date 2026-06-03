from transformers import pipeline 
class IntentRouter:
    def __init__(self):
        self.classifier= pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device="cpu"
        )
        self.labels=["asking a question about rules, documents, or eligibility",
            "applying, registering, submitting, or filling a form"]
    def route(self , text:str) ->str:
        result = self.classifier (text,candidate_labels = self.labels)
        top_label = result["labels"][0]
        score = result["scores"][0]
        print(f"Intent : {top_label} , Score : {score:.2f}")
        if "asking a question" in top_label:
            print("rag")
        else :
            print("apply")

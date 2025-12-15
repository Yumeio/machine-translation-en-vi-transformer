import os
import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash"

class GeminiScore:
    def __init__(self, api_key=None, model_name=MODEL_NAME):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found. Gemini Score will not work.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)

    def compute(self, predictions, references, sources=None):
        if not self.api_key:
            return 0.0
        
        scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            src_text = sources[i] if sources else ""
            prompt = f"""
            You are a professional translator evaluator.
            Please evaluate the quality of the following translation from English to Vietnamese.
            
            Source Code (English): {src_text}
            Reference Translation (Vietnamese): {ref}
            Candidate Translation (Vietnamese): {pred}
            
            Score the translation on a scale from 0 to 100 based on accuracy, fluency, and preservation of meaning.
            Return ONLY the number.
            """
            try:
                response = self.model.generate_content(prompt)
                score = float(response.text.strip())
                scores.append(score)
            except Exception as e:
                print(f"Error computing Gemini score: {e}")
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
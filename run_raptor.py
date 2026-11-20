import os
from pathlib import Path
from raptor import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    BaseEmbeddingModel,
    BaseSummarizationModel,
    BaseQAModel,
)
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1. Local Embedding Model
class LocalEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text: str) -> list:
        return self.model.encode(text).tolist()

# 2. Local Summarization Model (using a lightweight Hugging Face model)
class LocalSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        # Generate a concise summary of the context nodes
        # Adjust input length to prevent token overflow
        max_len = min(max_tokens, 130)
        min_len = max(30, max_len // 2)
        
        try:
            summary = self.summarizer(
                context, 
                max_length=max_len, 
                min_length=min_len, 
                do_sample=False
            )[0]['summary_text']
            return summary
        except Exception as e:
            # Fallback if text is too long or model hits limits
            return context[:500]

# 3. Local QA Model (using a Hugging Face text-generation or question-answering pipeline)
class LocalQAModel(BaseQAModel):
    def __init__(self, model_name="google/flan-t5-small"):
        self.qa_pipeline = pipeline("text2text-generation", model=model_name)

    def answer_question(self, context: str, question: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        # Generate answer locally
        result = self.qa_pipeline(prompt, max_length=128, do_sample=False)
        return result[0]['generated_text']

# 4. Combine everything into RAPTOR Configuration
config = RetrievalAugmentationConfig(
    embedding_model=LocalEmbeddingModel(),
    summarization_model=LocalSummarizationModel(),
    qa_model=LocalQAModel()
)

# 5. Load sample document
sample_path = Path("demo/sample.txt")
if not sample_path.exists():
    sample_path = Path("sample.txt")

text = sample_path.read_text(encoding="utf-8")
print(f"Loaded {len(text)} characters from {sample_path}")

# 6. Initialize and Build Tree Locally
print("Initializing RAPTOR with fully local models...")
ra = RetrievalAugmentation(config=config)

print("Building the RAPTOR tree locally (this will download local models on first run)...")
ra.add_documents(text)
print("RAPTOR tree construction finished successfully.")

# 7. Ask a question
question = "How did Cinderella reach her happy ending?"
answer = ra.answer_question(question=question)

print("\n--- Results ---")
print(f"Question: {question}")
print(f"Answer: {answer}")

# 8. Save tree
ra.save("demo/cinderella_local")
print("\nTree saved to demo/cinderella_local")
import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pinecone"
NAMESPACE = "global"

def inspect_all():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    dummy_vector = [0.0] * 768
    
    print("Querying Pinecone for all vectors in namespace 'global'...")
    results = index.query(
        namespace=NAMESPACE,
        vector=dummy_vector,
        top_k=100,
        include_metadata=True
    )
    
    summary = {}
    for match in results.matches:
        src = match.metadata.get("source_file") or match.metadata.get("source")
        if src not in summary:
            summary[src] = []
        summary[src].append({
            "id": match.id,
            "text_preview": (match.metadata.get("text") or "")[:100].replace("\n", " "),
            "page": match.metadata.get("page")
        })
    
    with open("tmp/all_pinecone_vectors.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Summary saved to tmp/all_pinecone_vectors.json. Found vectors for {len(summary)} sources.")

if __name__ == "__main__":
    inspect_all()

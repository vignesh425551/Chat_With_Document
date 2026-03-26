import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pinecone"
NAMESPACE = "global"

def search_text(search_str):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    dummy_vector = [0.0] * 768
    
    print(f"Searching Pinecone for string: '{search_str}'")
    results = index.query(
        namespace=NAMESPACE,
        vector=dummy_vector,
        top_k=100,
        include_metadata=True
    )
    
    matches = []
    for m in results.matches:
        text = m.metadata.get("text", "")
        if search_str.lower() in text.lower():
            matches.append({
                "id": m.id,
                "src": m.metadata.get("source_file"),
                "text": text
            })
            
    print(f"Found {len(matches)} matches.")
    for m in matches:
        print(f"Source: {m['src']}")
        print(f"ID: {m['id']}")
        print(f"Text: {m['text'][:200]}...")
        print("-" * 40)

if __name__ == "__main__":
    search_text("$10,000.00")
    search_text("John Doe")

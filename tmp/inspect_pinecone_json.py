import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pinecone"
NAMESPACE = "global"

def inspect():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    doc_name = "Invoice-BAI46CEW-0003.pdf"
    dummy_vector = [0.0] * 768
    
    print(f"Querying Pinecone for source_file: {doc_name}")
    results = index.query(
        namespace=NAMESPACE,
        vector=dummy_vector,
        filter={"source_file": {"$eq": doc_name}},
        top_k=50,
        include_metadata=True
    )
    
    output_data = {
        "doc_name": doc_name,
        "matches_count": len(results.matches),
        "matches": []
    }
    
    for match in results.matches:
        output_data["matches"].append({
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata
        })
    
    with open("tmp/pinecone_dump.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Results saved to tmp/pinecone_dump.json. Found {len(results.matches)} matches.")

if __name__ == "__main__":
    inspect()

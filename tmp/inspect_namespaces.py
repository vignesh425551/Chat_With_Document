import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pinecone"

def inspect_namespaces():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    
    dummy_vector = [0.0] * 768
    
    for ns in stats.namespaces:
        print(f"Querying namespace: '{ns}'")
        results = index.query(
            namespace=ns,
            vector=dummy_vector,
            top_k=50,
            include_metadata=True
        )
        
        matches = []
        for match in results.matches:
            matches.append({
                "src": match.metadata.get("source_file") or match.metadata.get("source"),
                "text": (match.metadata.get("text") or "")[:100],
                "id": match.id
            })
        
        with open(f"tmp/ns_{ns or 'default'}.json", "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=4)

if __name__ == "__main__":
    inspect_namespaces()

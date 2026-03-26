import os
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
        top_k=5,
        include_metadata=True
    )
    
    if not results.matches:
        print("No matches found for this document.")
        # Try a broader query to see what source_files are available
        print("\nBroad query to find available source_files:")
        broad_results = index.query(
            namespace=NAMESPACE,
            vector=dummy_vector,
            top_k=10,
            include_metadata=True
        )
        sources = set()
        for m in broad_results.matches:
            sources.add(m.metadata.get("source_file"))
        print(f"Available sources in recent queries: {sources}")
        return

    print(f"Found {len(results.matches)} matches.")
    for i, match in enumerate(results.matches):
        print(f"\n--- Result {i+1} ---")
        print(f"ID: {match.id}")
        print(f"Score: {match.score}")
        print(f"Metadata: {match.metadata}")
        text = match.metadata.get("text") or "NO TEXT FIELD"
        print(f"Text: {text}")
        print("-" * 50)

if __name__ == "__main__":
    inspect()

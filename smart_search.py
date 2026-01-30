import os

# CRITICAL: Set offline mode BEFORE any other imports
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Lazy load components (don't load at import time)
_embedder = None
_client = None
_collection = None

def get_components():
    """Lazy load ChromaDB and embedder only when needed"""
    global _embedder, _client, _collection
    
    if _embedder is None:
        print("Loading search components...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        
        _client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="./chroma_db"
        ))
        _collection = _client.get_collection("repair_guides")
        print(f"Loaded! Total guides: {_collection.count()}\n")
    
    return _embedder, _client, _collection

def search_all_guides(query, top_k=3):
    """Search across all repair guides"""
    _, _, collection = get_components()
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results

def search_by_brand_model(query, brand=None, model=None, top_k=20):
    """
    Search guides and filter by brand/model in post-processing.
    We search for more results (top_k=20) then filter to get top 3 relevant ones.
    """
    _, _, collection = get_components()
    
    # Get more results than needed
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # Filter results by brand/model
    filtered_docs = []
    filtered_metas = []
    filtered_distances = []
    
    if results['documents'] and results['documents'][0]:
        for i, metadata in enumerate(results['metadatas'][0]):
            title = metadata.get('title', '').lower()
            category = metadata.get('category', '').lower()
            
            # Check if brand matches
            brand_match = True
            if brand:
                brand_lower = brand.lower()
                brand_match = brand_lower in title or brand_lower in category
            
            # Check if model matches
            model_match = True
            if model:
                model_lower = model.lower()
                model_match = model_lower in title or model_lower in category
            
            # If both conditions met, keep this result
            if brand_match and model_match:
                filtered_docs.append(results['documents'][0][i])
                filtered_metas.append(metadata)
                filtered_distances.append(results['distances'][0][i] if results['distances'] else 0)
                
                # Stop when we have 3 good matches
                if len(filtered_docs) >= 3:
                    break
    
    # Return in same format as original query
    return {
        'documents': [filtered_docs],
        'metadatas': [filtered_metas],
        'distances': [filtered_distances],
        'ids': [[]] 
    }

def get_guide_count_by_brand(brand):
    """Count how many guides exist for a specific brand"""
    _, _, collection = get_components()
    
    # Get a large sample
    results = collection.query(
        query_texts=[brand],
        n_results=1000
    )
    
    count = 0
    if results['metadatas'] and results['metadatas'][0]:
        brand_lower = brand.lower()
        for metadata in results['metadatas'][0]:
            title = metadata.get('title', '').lower()
            category = metadata.get('category', '').lower()
            if brand_lower in title or brand_lower in category:
                count += 1
    
    return count

# Test code
if __name__ == "__main__":
    print("="*60)
    print("Testing Smart Search (Offline Mode)")
    print("="*60)
    
    # Test 1: Search all guides
    print("\n1. Searching ALL guides for 'washing machine drain':")
    results = search_all_guides("washing machine drain", top_k=3)
    for i, metadata in enumerate(results['metadatas'][0]):
        print(f"   {i+1}. {metadata['title']}")
        print(f"      Category: {metadata['category']}")
    
    # Test 2: Search Kenmore only
    print("\n2. Searching KENMORE guides for 'washing machine drain':")
    results = search_by_brand_model("washing machine drain", brand="Kenmore", top_k=20)
    if results['metadatas'][0]:
        for i, metadata in enumerate(results['metadatas'][0]):
            print(f"   {i+1}. {metadata['title']}")
            print(f"      Category: {metadata['category']}")
    else:
        print("   No Kenmore guides found")
    
    # Test 3: Search Kenmore Elite HE3 specifically
    print("\n3. Searching KENMORE ELITE HE3 for 'drain problem':")
    results = search_by_brand_model("drain problem", brand="Kenmore Elite", model="HE3", top_k=20)
    if results['metadatas'][0]:
        for i, metadata in enumerate(results['metadatas'][0]):
            print(f"   {i+1}. {metadata['title']}")
            print(f"      Category: {metadata['category']}")
    else:
        print("   No Kenmore Elite HE3 guides found")
    
    # Test 4: Count guides per brand
    print("\n4. Estimated guide counts by brand:")
    brands = ["Kenmore", "Samsung", "Bosch", "iPhone", "Mac"]
    for brand in brands:
        count = get_guide_count_by_brand(brand)
        print(f"   {brand}: ~{count} guides")
    
    print("\n" + "="*60)
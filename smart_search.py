import os
import re

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


def extract_device_info_enhanced(query: str) -> dict:
    """Extract brand/model/component with a confidence score."""
    known_brands = [
        'Canon', 'Samsung', 'iPhone', 'Kenmore', 'Olympus', 'Sony', 'LG',
        'Whirlpool', 'Bosch', 'GE', 'Nikon', 'Apple', 'Dell', 'HP', 'Lenovo',
        'Asus', 'Maytag', 'Frigidaire', 'KitchenAid', 'Electrolux', 'Panasonic'
    ]

    patterns = {
        'alphanumeric': r'\b[A-Z]+[-\s]?\d+[A-Z\d-]*\b',
        'numeric_only': r'\b\d{4,}\b',
        'series': r'\b(?:Elite\s+)?[A-Z]+\s*\d+\s*[A-Z\d-]*\b',
    }

    components = {
        'screen': ['screen', 'display', 'lcd', 'panel'],
        'battery': ['battery', 'power'],
        'sensor': ['sensor', 'image sensor'],
        'faceplate': ['faceplate', 'face plate', 'front panel'],
        'lens': ['lens', 'zoom'],
        'pump': ['pump', 'drain'],
        'motor': ['motor'],
    }

    result = {
        'brand': None,
        'model': None,
        'component': None,
        'extraction_confidence': 0.0,
        'query_type': 'unclear'
    }

    query_lower = query.lower()
    for brand in known_brands:
        if brand.lower() in query_lower:
            result['brand'] = brand
            result['extraction_confidence'] += 0.4
            break

    for pattern in patterns.values():
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            result['model'] = matches[0].strip()
            result['extraction_confidence'] += 0.4
            break

    for component_type, keywords in components.items():
        if any(keyword in query_lower for keyword in keywords):
            result['component'] = component_type
            result['extraction_confidence'] += 0.2
            break

    if result['brand'] and result['model']:
        result['query_type'] = 'specific'
    elif result['brand'] or result['component']:
        result['query_type'] = 'general'

    return result


def score_guide_match(query_info: dict, guide_metadata: dict) -> int:
    """Score guide relevance on a 0-100 scale."""
    score = 0
    guide_title_lower = guide_metadata.get('title', '').lower()
    guide_category_lower = guide_metadata.get('category', '').lower()

    if query_info.get('brand'):
        brand_lower = query_info['brand'].lower()
        if brand_lower in guide_title_lower or brand_lower in guide_category_lower:
            score += 40
        else:
            return 0

    if query_info.get('model'):
        model_lower = query_info['model'].lower()
        if model_lower in guide_title_lower or model_lower in guide_category_lower:
            score += 40
        else:
            model_normalized = re.sub(r'[-\s]', '', model_lower)
            title_normalized = re.sub(r'[-\s]', '', guide_title_lower)
            category_normalized = re.sub(r'[-\s]', '', guide_category_lower)
            if model_normalized and (
                model_normalized in title_normalized or model_normalized in category_normalized
            ):
                score += 20

    if query_info.get('component'):
        component_lower = query_info['component'].lower()
        if component_lower in guide_title_lower:
            score += 15
        if component_lower in guide_category_lower:
            score += 5

    return min(score, 100)


def multi_stage_search(query: str, query_info: dict) -> dict:
    """Run exact -> brand -> semantic retrieval with scored results."""
    search_result = {
        'results': [],
        'confidence_level': 'none',
        'stage_used': None,
        'total_candidates': 0,
    }

    # Stage 1: exact brand + model
    if query_info.get('brand') and query_info.get('model'):
        stage1 = search_by_brand_model(
            f"{query_info['brand']} {query_info['model']}",
            brand=query_info['brand'],
            model=query_info['model'],
            top_k=12,
        )
        scored = []
        for i, doc in enumerate(stage1.get('documents', [[]])[0]):
            metadata = stage1['metadatas'][0][i]
            score = score_guide_match(query_info, metadata)
            if score >= 75:
                scored.append({'document': doc, 'metadata': metadata, 'score': score, 'rank': i + 1})
        if scored:
            scored.sort(key=lambda x: x['score'], reverse=True)
            search_result.update({
                'results': scored,
                'confidence_level': 'high',
                'stage_used': 1,
                'total_candidates': len(scored),
            })
            return search_result

    # Stage 2: brand filtered
    if query_info.get('brand'):
        stage2 = search_by_brand_model(
            f"{query_info['brand']} {query_info.get('component') or 'repair'}",
            brand=query_info['brand'],
            model=None,
            top_k=24,
        )
        scored = []
        for i, doc in enumerate(stage2.get('documents', [[]])[0]):
            metadata = stage2['metadatas'][0][i]
            score = score_guide_match(query_info, metadata)
            if score >= 50:
                scored.append({'document': doc, 'metadata': metadata, 'score': score, 'rank': i + 1})
        if scored:
            scored.sort(key=lambda x: x['score'], reverse=True)
            search_result.update({
                'results': scored,
                'confidence_level': 'medium',
                'stage_used': 2,
                'total_candidates': len(scored),
            })
            return search_result

    # Stage 3: semantic fallback
    stage3 = search_all_guides(query, top_k=12)
    scored = []
    for i, doc in enumerate(stage3.get('documents', [[]])[0]):
        metadata = stage3['metadatas'][0][i]
        score = score_guide_match(query_info, metadata)
        if score >= 30:
            scored.append({'document': doc, 'metadata': metadata, 'score': score, 'rank': i + 1})

    if scored:
        scored.sort(key=lambda x: x['score'], reverse=True)
        search_result.update({
            'results': scored,
            'confidence_level': 'low',
            'stage_used': 3,
            'total_candidates': len(scored),
        })

    return search_result


def filter_results_by_confidence(search_results: dict, min_score: int = 70) -> list:
    """Return top 3 results above threshold."""
    results = search_results.get('results', [])
    if not results:
        return []
    filtered = [result for result in results if result['score'] >= min_score]
    return filtered[:3]

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

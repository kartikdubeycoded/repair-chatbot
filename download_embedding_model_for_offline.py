"""
Run this script ONCE with internet connection to cache the embedding model.
After this, your system will work completely offline.
"""

from sentence_transformers import SentenceTransformer
import os

print("="*60)
print("Pre-downloading Embedding Model for Offline Use")
print("="*60)
print("\nThis needs to be run ONCE with internet connection.")
print("After this, the chatbot will work completely offline.\n")

# Don't set offline mode for this script
if 'HF_HUB_OFFLINE' in os.environ:
    del os.environ['HF_HUB_OFFLINE']
if 'TRANSFORMERS_OFFLINE' in os.environ:
    del os.environ['TRANSFORMERS_OFFLINE']

try:
    print("Downloading 'all-MiniLM-L6-v2' model (~80MB)...")
    print("This may take a few minutes...\n")
    
    # Download and cache the model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test it works
    print("Testing model...")
    test_embedding = embedder.encode("test sentence")
    print(f"Model loaded successfully! Embedding size: {len(test_embedding)}")
    
    # Find cache location
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    print(f"\nModel cached at: {cache_dir}")
    
    print("\n" + "="*60)
    print("SUCCESS! Model downloaded and cached.")
    print("Your chatbot will now work completely offline!")
    print("="*60)
    
except Exception as e:
    print(f"\nError: {e}")
    print("\nMake sure you have internet connection and try again.")
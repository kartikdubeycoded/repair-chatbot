import os
import sqlite3
import time

# CRITICAL: Set offline mode FIRST before any AI library imports
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import gradio as gr
from ctransformers import AutoModelForCausalLM

# Import our helper functions
from context_manager import (
    is_vague_question,
    generate_clarification_prompt,
    extract_brand_and_model,
    get_last_user_messages,
)
from smart_search import search_all_guides, search_by_brand_model

print("Starting Smart Repair Chatbot (Non-llama.cpp backend) with Streaming...\n")

# Initialize database
def init_database():
    conn = sqlite3.connect('conversation_logs.db')
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_question TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            sources TEXT,
            response_time_seconds REAL,
            context_info TEXT
        )
    '''
    )
    conn.commit()
    conn.close()


init_database()


# Load AI model (no llama_cpp)
def load_model():
    """Load GGUF model using ctransformers instead of llama_cpp."""
    model_path = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Check the exact filename/path inside ./models and ensure the .gguf file exists."
        )

    print("Loading Qwen model with ctransformers...")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="qwen2",
        context_length=4096,
        gpu_layers=0,
        threads=max(1, os.cpu_count() or 4),
    )


llm = load_model()
print("✓ All components loaded!\n")


# Conversation state (shared across calls)
conversation_state = {
    "active_brand": None,
    "active_model": None,
    "active_guide_locked": False,
    "language": "English",
}


def update_language(lang):
    """Update preferred reply language for the active session."""
    conversation_state["language"] = lang


def log_conversation(question, response, sources, response_time, context_info):
    """Save conversation to database"""
    conn = sqlite3.connect('conversation_logs.db')
    cursor = conn.cursor()
    sources_str = " | ".join(sources) if sources else "None"
    cursor.execute(
        '''
        INSERT INTO conversations (user_question, bot_response, sources, response_time_seconds, context_info)
        VALUES (?, ?, ?, ?, ?)
    ''',
        (question, response, sources_str, response_time, context_info),
    )
    conn.commit()
    conn.close()


def stream_response(prompt):
    """Token streaming wrapper for ctransformers."""
    for token in llm(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        max_new_tokens=300,
        temperature=0.3,
        stop=["<|im_end|>", "<|im_start|>"],
        stream=True,
    ):
        yield token


def chat(message, history):
    """Smart chat with streaming response"""
    global conversation_state

    start_time = time.time()
    reply_language = conversation_state.get("language", "English")

    # Step 1: Check if this is a vague first question
    is_vague, reason = is_vague_question(message, history)

    if is_vague:
        response = generate_clarification_prompt(message, reason)
        response_time = time.time() - start_time
        log_conversation(message, response, [], response_time, "clarification_requested")
        yield response
        return

    # Step 2: Try to extract brand/model from current message
    extracted = extract_brand_and_model(message)

    if extracted['brand']:
        conversation_state["active_brand"] = extracted['brand']
        conversation_state["active_model"] = extracted['model']
        conversation_state["active_guide_locked"] = True

    # Step 3: Also check previous messages for brand/model if not found
    if not conversation_state["active_brand"] and history:
        last_messages = get_last_user_messages(history, count=2)
        for prev_msg in last_messages:
            prev_extracted = extract_brand_and_model(prev_msg)
            if prev_extracted['brand']:
                conversation_state["active_brand"] = prev_extracted['brand']
                conversation_state["active_model"] = prev_extracted['model']
                break

    # Step 4: Search for repair guides
    brand = conversation_state["active_brand"]
    model = conversation_state["active_model"]

    if brand or model:
        search_results = search_by_brand_model(message, brand=brand, model=model, top_k=20)
        context_info = f"filtered:{brand}_{model}"
    else:
        search_results = search_all_guides(message, top_k=3)
        context_info = "general_search"

    # Step 5: Build context
    context = ""
    sources = []

    if search_results['metadatas'] and search_results['metadatas'][0]:
        for i, doc in enumerate(search_results['documents'][0]):
            metadata = search_results['metadatas'][0][i]
            doc_short = doc[:800] + "..." if len(doc) > 800 else doc
            context += f"\n--- Guide {i+1}: {metadata['title']} ---\n{doc_short}\n"
            sources.append(metadata['title'])

    # Step 6: Create prompt
    language_instruction = (
        "Respond in Dutch. Keep technical terms clear and practical for a Dutch-speaking repair engineer."
        if reply_language == "Dutch"
        else "Respond in English."
    )

    if context:
        prompt = f"""You are a repair assistant. Answer based on these repair guides:

{context}

Question: {message}
{language_instruction}
Provide a clear, concise answer with specific steps:"""
    else:
        prompt = f"""Question: {message}
{language_instruction}
Provide brief general repair advice:"""

    # Step 7: Generate response with STREAMING
    full_response = ""

    # Add brand/model header if available
    header = ""
    if brand or model:
        brand_model_text = f"{brand} {model}" if model else brand
        header = f"**[{brand_model_text}]**\n\n"
        full_response = header
        yield full_response

    for token in stream_response(prompt):
        full_response += token
        yield full_response

    # Step 8: Add sources and timing after generation completes
    response_time = time.time() - start_time

    footer = ""
    if sources:
        footer += f"\n\n** Sources:** {sources[0]}"
        if len(sources) > 1:
            footer += f" (+{len(sources)-1} more)"

    footer += f"\n\n_ {response_time:.1f}s_"

    full_response += footer
    yield full_response

    # Step 9: Log conversation
    log_conversation(message, full_response, sources, response_time, context_info)


with gr.Blocks(title="🔧 Smart Repair Assistant (Streaming, non-llama.cpp)") as demo:
    gr.Markdown("# 🔧 Smart Repair Assistant (Streaming)")
    gr.Markdown(
        """
        **AI repair assistant with live streaming responses (without llama_cpp)**

        - Tell me your appliance brand/model for faster, more accurate help!
        - I remember context across the conversation
        - Responses stream in real-time as they're generated
        - 18,995 repair guides
        """
    )

    language = gr.Radio(
        choices=["English", "Dutch"],
        value="English",
        label="Reply language / Antwoordtaal",
    )
    language.change(update_language, language, None)

    gr.ChatInterface(
        fn=chat,
        examples=[
            "My washing machine is broken",
            "My Kenmore Elite HE3 won't drain",
            "Samsung refrigerator not cooling",
            "Mijn wasmachine pompt geen water af",
        ],
    )


if __name__ == "__main__":
    print("=" * 60)
    print(" Starting Smart Chatbot with Streaming (non-llama.cpp)...")
    print("=" * 60)
    demo.launch(share=False, inbrowser=True)

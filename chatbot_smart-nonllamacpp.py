import os
import sqlite3
import threading
import time

# CRITICAL: Set offline mode FIRST before any AI library imports
# Keep these enabled when you've already downloaded model files locally.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Import our helper functions
from context_manager import (
    is_vague_question,
    generate_clarification_prompt,
    extract_brand_and_model,
    get_last_user_messages,
)
from smart_search import search_all_guides, search_by_brand_model

print("Starting Smart Repair Chatbot (Transformers backend) with Streaming...\n")


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


# Load AI model (no llama_cpp, no ctransformers)
def load_model_and_tokenizer():
    """Load local GPTQ model with transformers."""
    model_dir = "./models/qwen-gptq"

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found at {model_dir}. "
            "Download model files first into ./models/qwen-gptq."
        )

    required_files = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required_files if not os.path.exists(os.path.join(model_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing required model metadata files in {model_dir}: {missing}. "
            "This looks like a file/folder issue (incomplete download or wrong folder)."
        )

    print("Loading Qwen GPTQ model with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()
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
    """Token streaming wrapper for transformers."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": 300,
        "temperature": 0.3,
        "do_sample": True,
        "streamer": streamer,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token_text in streamer:
        yield token_text

    thread.join()


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
    model_name = conversation_state["active_model"]

    if brand or model_name:
        search_results = search_by_brand_model(message, brand=brand, model=model_name, top_k=20)
        context_info = f"filtered:{brand}_{model_name}"
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
    if brand or model_name:
        brand_model_text = f"{brand} {model_name}" if model_name else brand
        full_response = f"**[{brand_model_text}]**\n\n"
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
    print(" Starting Smart Chatbot with Streaming (transformers GPTQ)...")
    print("=" * 60)
    demo.launch(share=False, inbrowser=True)

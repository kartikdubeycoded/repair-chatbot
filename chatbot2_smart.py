import os
import sqlite3
import threading
import time

# Optional offline mode: set REPAIR_CHATBOT_OFFLINE=1 after model download.
if os.getenv("REPAIR_CHATBOT_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from context_manager import (
    is_vague_question,
    generate_clarification_prompt,
    extract_brand_and_model,
    get_last_user_messages,
)
from smart_search import search_all_guides, search_by_brand_model

MODEL_DIR = "./models2"
DB_PATH = "conversation_logs.db"

print("Starting Smart Repair Chatbot (Transformers backend, non-llama.cpp)...\n")


def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_question TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            sources TEXT,
            response_time_seconds REAL,
            context_info TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_conversation(question, response, sources, response_time, context_info):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sources_str = " | ".join(sources) if sources else "None"
    cursor.execute(
        """
        INSERT INTO conversations (user_question, bot_response, sources, response_time_seconds, context_info)
        VALUES (?, ?, ?, ?, ?)
        """,
        (question, response, sources_str, response_time, context_info),
    )
    conn.commit()
    conn.close()


def load_model_and_tokenizer():
    """Load NON-GPTQ local model from ./models2."""
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Put your NON-GPTQ model files in ./models2"
        )

    required_files = ["config.json", "tokenizer_config.json"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {MODEL_DIR}: {missing}. "
            "Expected at least config.json and tokenizer_config.json"
        )

    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read().lower()

    if '"quantization_config"' in config_text and '"gptq"' in config_text:
        raise RuntimeError(
            "Detected GPTQ model in ./models2. This script is for NON-GPTQ models.\n"
            "Use a non-GPTQ model (for example: Qwen/Qwen2.5-1.5B-Instruct)."
        )

    print(f"Loading NON-GPTQ local model from: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    return model, tokenizer


def stream_response(prompt):
    """Stream generated tokens from the model."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful repair assistant. Keep answers practical and step-by-step.",
        },
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        **model_inputs,
        "max_new_tokens": 320,
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


# Shared conversation state
conversation_state = {
    "active_brand": None,
    "active_model": None,
    "language": "English",
}


def update_language(lang):
    conversation_state["language"] = lang


def build_search_context(message, history):
    extracted = extract_brand_and_model(message)

    if extracted["brand"]:
        conversation_state["active_brand"] = extracted["brand"]
        conversation_state["active_model"] = extracted["model"]

    if not conversation_state["active_brand"] and history:
        last_messages = get_last_user_messages(history, count=2)
        for prev_msg in last_messages:
            prev_extracted = extract_brand_and_model(prev_msg)
            if prev_extracted["brand"]:
                conversation_state["active_brand"] = prev_extracted["brand"]
                conversation_state["active_model"] = prev_extracted["model"]
                break

    brand = conversation_state["active_brand"]
    model_name = conversation_state["active_model"]

    if brand or model_name:
        search_results = search_by_brand_model(message, brand=brand, model=model_name, top_k=20)
        context_info = f"filtered:{brand}_{model_name}"
    else:
        search_results = search_all_guides(message, top_k=3)
        context_info = "general_search"

    context = ""
    sources = []

    if search_results.get("metadatas") and search_results["metadatas"][0]:
        for i, doc in enumerate(search_results["documents"][0]):
            metadata = search_results["metadatas"][0][i]
            short_doc = doc[:900] + "..." if len(doc) > 900 else doc
            context += f"\n--- Guide {i+1}: {metadata['title']} ---\n{short_doc}\n"
            sources.append(metadata["title"])

    display_brand_model = None
    if brand and model_name:
        display_brand_model = f"{brand} {model_name}"
    elif brand:
        display_brand_model = brand
    elif model_name:
        display_brand_model = model_name

    return context, sources, context_info, display_brand_model


def chat(message, history):
    start_time = time.time()
    reply_language = conversation_state.get("language", "English")

    is_vague, reason = is_vague_question(message, history)
    if is_vague:
        response = generate_clarification_prompt(message, reason)
        response_time = time.time() - start_time
        log_conversation(message, response, [], response_time, "clarification_requested")
        yield response
        return

    context, sources, context_info, display_brand_model = build_search_context(message, history)

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
Provide a clear, concise answer with specific steps."""
    else:
        prompt = f"""Question: {message}
{language_instruction}
Provide brief general repair advice."""

    full_response = ""

    if display_brand_model:
        full_response = f"**[{display_brand_model}]**\n\n"
        yield full_response

    for token in stream_response(prompt):
        full_response += token
        yield full_response

    response_time = time.time() - start_time

    footer = ""
    if sources:
        footer += f"\n\n**Sources:** {sources[0]}"
        if len(sources) > 1:
            footer += f" (+{len(sources)-1} more)"

    footer += f"\n\n_{response_time:.1f}s_"

    full_response += footer
    yield full_response

    log_conversation(message, full_response, sources, response_time, context_info)


# Boot
init_database()
model, tokenizer = load_model_and_tokenizer()
print("✓ All components loaded!\n")

with gr.Blocks(title="🔧 Smart Repair Assistant (transformers)") as demo:
    gr.Markdown("# 🔧 Smart Repair Assistant")
    gr.Markdown(
        """
        **AI repair assistant with live streaming responses (Transformers backend)**

        - Tell me your appliance brand/model for faster, more accurate help
        - Context-aware across conversation
        - English / Dutch reply support
        - Real-time streamed generation
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
    print(" Starting Smart Chatbot with Streaming (transformers, NON-GPTQ)...")
    print("=" * 60)

    demo.launch(
        share=False,
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True,
    )
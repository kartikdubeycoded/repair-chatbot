"""Main entrypoint for Repair Chatbot Pro."""

import os
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from chatbot_ui import create_chatbot_ui
from context_manager import (
    classify_query_type,
    extract_brand_and_model,
    generate_alternative_options,
    handle_greeting,
    validate_response_quality,
)
from model_router import ModelRouter
from retrieval_validator import validator
from smart_search import (
    extract_device_info_enhanced,
    filter_results_by_confidence,
    multi_stage_search,
)


conversation_state = {
    'language': 'English',
    'last_options': [],
    'last_query_info': None,
}


def set_language(language: str):
    conversation_state['language'] = language


def _history_to_records(history):
    records = []
    for user, assistant in history:
        records.append({'role': 'user', 'content': [{'text': user}]})
        records.append({'role': 'assistant', 'content': [{'text': assistant}]})
    return records


def _build_prompt(message: str, top_sources: list, language: str) -> str:
    if not top_sources:
        if language == 'Dutch':
            return (
                f"Gebruikersvraag: {message}\n"
                "Er zijn geen betrouwbare bronnen gevonden. Geef aan dat er geen geschikte gids beschikbaar is."
            )
        return (
            f"User question: {message}\n"
            "No reliable guide was found. Clearly say no exact guide is available and ask for model/component details."
        )

    snippets = []
    for i, source in enumerate(top_sources, 1):
        title = source['metadata'].get('title', 'Unknown')
        doc = source['document'][:1200]
        snippets.append(f"Guide {i}: {title}\n{doc}")

    joined = "\n\n".join(snippets)

    if language == 'Dutch':
        return (
            "Je bent een reparatie-assistent. Gebruik alleen de broninformatie hieronder. "
            "Noem het apparaatsmodel expliciet en geef duidelijke stappen.\n\n"
            f"Bronnen:\n{joined}\n\nVraag: {message}"
        )

    return (
        "You are a repair assistant. Use only the source guides below. "
        "Mention the device model explicitly and provide clear, grounded steps.\n\n"
        f"Sources:\n{joined}\n\nQuestion: {message}"
    )


def chat(message, history):
    start = time.time()
    language = conversation_state['language']
    records = _history_to_records(history)
    query_type = classify_query_type(message, records)

    if query_type == 'greeting':
        yield handle_greeting(language)
        return

    if query_type == 'closing':
        yield 'You are welcome! 👋' if language == 'English' else 'Graag gedaan! 👋'
        return

    if query_type == 'device_selection' and conversation_state['last_options']:
        selection = None
        stripped = message.strip()
        if stripped.isdigit():
            selection = int(stripped) - 1
        if selection is not None and 0 <= selection < len(conversation_state['last_options']):
            picked = conversation_state['last_options'][selection]
            title = picked['metadata'].get('title', 'Unknown')
            prompt = _build_prompt(title, [picked], language)
            full = ''
            for token in router.route_and_generate(prompt, language=language, max_tokens=300):
                full += token
                yield full
            return

    query_info = extract_device_info_enhanced(message)
    if not query_info.get('brand'):
        basic = extract_brand_and_model(message)
        if basic.get('brand'):
            query_info['brand'] = basic['brand']
            query_info['model'] = query_info.get('model') or basic.get('model')

    search_results = multi_stage_search(message, query_info)
    filtered = filter_results_by_confidence(search_results, min_score=70)

    conversation_state['last_options'] = search_results.get('results', [])[:5]
    conversation_state['last_query_info'] = query_info

    confidence = search_results.get('confidence_level')
    if confidence == 'medium' and not filtered:
        response = generate_alternative_options(query_info, search_results.get('results', []), language=language)
        validation = validate_response_quality(response, query_info, [])
        validator.log_retrieval_performance(message, search_results, validation)
        yield response
        return

    if not filtered:
        response = generate_alternative_options(query_info, [], language=language)
        validation = validate_response_quality(response, query_info, [])
        validator.log_retrieval_performance(message, search_results, validation)
        yield response
        return

    prompt = _build_prompt(message, filtered, language)
    full_response = ''
    for token in router.route_and_generate(prompt, language=language, max_tokens=380):
        full_response += token
        yield full_response

    validation = validate_response_quality(full_response, query_info, filtered)
    validator.log_retrieval_performance(message, search_results, validation)

    elapsed = time.time() - start
    footer = (
        f"\n\n---\nAccuracy: {validation['accuracy_score']:.0f}/100 | "
        f"Confidence: {confidence.upper()} | Time: {elapsed:.1f}s"
    )
    yield full_response + footer


router = ModelRouter(tinyllama_path='./models/tinyllama', qwen_path='./models2/qwen')
router.load_models()

status_parts = []
status_parts.append('TinyLlama: loaded' if router.tinyllama_model is not None else 'TinyLlama: fallback')
status_parts.append('Qwen: loaded' if router.qwen_model is not None else 'Qwen: fallback')
model_status = ' | '.join(status_parts)

if __name__ == '__main__':
    print('=' * 60)
    print('Starting Repair Chatbot Pro...')
    print('=' * 60)
    demo = create_chatbot_ui(chat, set_language, model_status)
    demo.launch(share=False, inbrowser=True)

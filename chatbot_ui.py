"""UI module for chatbot_pro using Gradio."""


def create_chatbot_ui(chat_fn, set_language_fn, model_status_text: str):
    """Create and return a modern Gradio interface."""
    import gradio as gr

    with gr.Blocks(title='🔧 Repair Chatbot Pro') as demo:
        gr.Markdown('# 🔧 Repair Chatbot Pro')
        gr.Markdown(
            'Dual-model repair assistant with validated retrieval, confidence routing, and accuracy scoring.'
        )

        with gr.Row():
            language = gr.Radio(
                choices=['English', 'Dutch'],
                value='English',
                label='Language / Taal',
                interactive=True,
            )
            gr.Textbox(
                value=model_status_text,
                label='Model Status',
                interactive=False,
            )

        language.change(set_language_fn, language, None)

        gr.ChatInterface(
            fn=chat_fn,
            examples=[
                'Hi',
                'Olympus D5900 faceplate replacement',
                'Samsung RF28R7201SR not cooling',
                'Ik wil de batterij van mijn iPhone vervangen',
            ],
        )

    return demo

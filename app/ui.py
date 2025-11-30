from __future__ import annotations

import gradio as gr

from app.pipelines.caption import caption_pipeline
from app.pipelines.chat import chat_pipeline
from app.config import config


def build_interface() -> gr.Blocks:
    custom_css = """
    /* Slightly enlarge overall UI scale */
    .gradio-container {
        font-size: 18px;
    }

    /* Shrink chat buttons: override Gradio's full-width style */
    .chat-button {
        flex: 0 0 auto !important;
        max-width: 150px !important;
    }

    .chat-button * {
        width: auto !important;
        padding: 0.3rem 0.8rem !important;
        font-size: 0.85rem !important;
    }
    """

    with gr.Blocks(title="LLM Multimodal WebUI", css=custom_css) as demo:
        gr.HTML(
            "<h1 style='text-align:center; font-size: 2rem;'>LLM Multimodal WebUI</h1>"
        )
        gr.Markdown(
            f"model: {config.model.model_id}"
        )

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=500)
                    chat_history = gr.State([])

                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Your message",
                            placeholder="Type a message and press Send",
                            lines=2,
                        )
                        send_btn = gr.Button(
                            "Send", variant="primary", elem_classes="chat-button"
                        )
                        clear_btn = gr.Button(
                            "Clear", elem_classes="chat-button"
                        )

                with gr.Column(scale=1, elem_id="gen-params"):
                    max_new_tokens = gr.Slider(
                        minimum=16,
                        maximum=2048,
                        value=config.model.max_new_tokens,
                        step=16,
                        label="Max new tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=config.model.temperature,
                        step=0.05,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=config.model.top_p,
                        step=0.05,
                        label="Top-p",
                    )

            send_btn.click(
                fn=chat_pipeline,
                inputs=[
                    chat_input,
                    chat_history,
                    max_new_tokens,
                    temperature,
                    top_p,
                ],
                outputs=[chatbot, chat_history, chat_input],
            )

            clear_btn.click(
                lambda: ([], [], ""),
                None,
                [chatbot, chat_history, chat_input],
                queue=False,
            )

        with gr.Tab("Image Captioning"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload image",
                        type="pil",
                    )
                    caption_prompt = gr.Textbox(
                        label="Caption prompt",
                        value="Describe this image in detail.",
                        lines=2,
                    )
                    caption_max_tokens = gr.Slider(
                        minimum=16,
                        maximum=512,
                        value=200,
                        step=16,
                        label="Max new tokens",
                    )
                    caption_btn = gr.Button("Generate caption", variant="primary")

                with gr.Column():
                    caption_output = gr.Textbox(
                        label="Caption",
                        lines=8,
                        interactive=False,
                    )

            caption_btn.click(
                fn=caption_pipeline,
                inputs=[image_input, caption_prompt, caption_max_tokens],
                outputs=[caption_output],
            )

        demo.queue(max_size=32)

    return demo

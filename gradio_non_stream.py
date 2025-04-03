import gradio as gr
from test import (
    async_bot_non_stream,
    UserMessage,
    SystemMessage,
    ChatHistory,
    ChatHistoryEntry,
)

list_llm = ["Gemma", "Qwen Coder", "Nemo", "Deepseek-R1"]


def llm(model):
    match model:
        case "Gemma":
            return "hf.co/google/gemma-3-4b-it-qat-q4_0-gguf:latest"
        case "Qwen Coder":
            return "qwen2.5-coder-7b"
        case "Nemo":
            return "mistral-nemo-12b-instruct"
        case "Deepseek-R1":
            return "deepseek-r1-8b-tools"


async def user(user_message, history):
    history.append({"role": "user", "content": user_message})
    return "", history


async def bot(msg, history, model, system_message):
    chat_history = ChatHistory()
    for entry in history:
        if entry["role"] == "user":
            chat_history.history.append(
                ChatHistoryEntry(user=UserMessage(content=entry["content"]))
            )
        elif entry["role"] == "assistant":
            chat_history.history[-1].assistant = entry["content"]

    user_message = UserMessage(content=msg)
    system_message_obj = SystemMessage(content=system_message)

    full_response = await async_bot_non_stream(
        user_message, chat_history, system_message_obj, model
    )
    history.append({"role": "assistant", "content": full_response})
    yield history


with gr.Blocks(title="Chati", fill_height=True, fill_width=True) as chati_interface:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                gr.HTML(
                    """
                <h1><center>Chatbot</center></h1> 
               """,
                    container=True,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row(elem_classes="tab_sub2"):
                        db = gr.Dropdown(
                            choices=list_llm, label="LLM Auswahl", value="Gemma"
                        )
                        model = gr.Textbox(visible=False)
                        system_message = gr.Textbox(
                            label="System Prompt",
                            value="You are Gemma, a very experienced and world-class  assistant. Your primary function is to answer questions. You will be provided with questions, and your task is to provide accurate and concise answers. Do not ask for clarification or engage in conversational filler. Respond directly to the question.",
                        )
                    with gr.Row():
                        chatbot = gr.Chatbot(show_copy_all_button=True, type="messages")
                    with gr.Row():
                        msg = gr.Textbox(label="Eingabe")
                    with gr.Row():
                        gr.Examples(
                            [
                                "Woraus besteht ein Bogen?",
                                "Nenne mir die Spaltungsprodukte von Methan und deren industrielle Verwendung.",
                                "Warum sind Tomaten kein Gemüse?",
                            ],
                            msg,
                            label="Beispiele",
                        )
                    with gr.Row():
                        senden = gr.Button("Senden", variant="primary")
                        clear = gr.ClearButton(
                            components=[msg, chatbot],
                            value="Löschen",
                            variant="secondary",
                        )

                    senden = (
                        senden.click(llm, db, model)
                        .then(user, [msg, chatbot], [msg, chatbot], queue=True)
                        .then(bot, [msg, chatbot, model, system_message], chatbot)
                    )
                    enter_submit = (
                        msg.submit(llm, db, model)
                        .then(user, [msg, chatbot], [msg, chatbot], queue=True)
                        .then(bot, [msg, chatbot, model, system_message], chatbot)
                    )

if __name__ == "__main__":
    chati_interface.queue()
    chati_interface.launch(server_name="0.0.0.0")

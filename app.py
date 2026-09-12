"""Gradio chat interface for the Agentic Profile Matching assistant.

Each browser tab gets its own conversation thread (a random ``thread_id``)
so the LangGraph agent's ``MemorySaver`` checkpointer keeps job requirements,
the candidate shortlist, and reasoning scoped to that session.
"""

import uuid

import gradio as gr

from matching_agent import chat

EXAMPLES_MARKDOWN = """
# Agentic Profile Matching

Chat with the recruiting assistant. Example flow:

1. "Find candidates with React and 3+ years experience."
2. "Compare the top 3 candidates."
3. "Why did the top candidate rank higher than the second candidate?"
4. "Make TypeScript a must-have and rerank."
5. "Generate interview questions for the top candidate."
"""


def new_thread_id() -> str:
    """Generate a fresh conversation thread id."""
    return str(uuid.uuid4())


def user_submit(message: str, history: list, thread_id: str):
    """Send ``message`` to the agent and append the exchange to ``history``."""
    if not message or not message.strip():
        return history, thread_id, ""

    reply = chat(message, thread_id)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, thread_id, ""


def start_new_conversation():
    """Reset the chat and start a brand-new conversation thread."""
    return [], new_thread_id()


def main() -> None:
    """Create and launch the Gradio interface."""
    with gr.Blocks(title="Agentic Profile Matching") as demo:
        gr.Markdown(EXAMPLES_MARKDOWN)

        thread_state = gr.State(new_thread_id)
        chatbot = gr.Chatbot(height=500, type="messages", label="Recruiting Assistant")
        msg = gr.Textbox(
            label="Message",
            placeholder="Describe a role, or ask about the current shortlist...",
        )
        new_chat_button = gr.Button("New conversation")

        msg.submit(
            fn=user_submit,
            inputs=[msg, chatbot, thread_state],
            outputs=[chatbot, thread_state, msg],
        )
        new_chat_button.click(
            fn=start_new_conversation,
            inputs=None,
            outputs=[chatbot, thread_state],
        )

    demo.launch()


if __name__ == "__main__":
    main()


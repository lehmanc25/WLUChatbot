import gradio as gr
from transformers import pipeline
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the model for generating text
model_name = "microsoft/phi-2"
model = pipeline("text-generation", model=model_name)

def generate_text(prompt):
    logging.info(f"Generating text for prompt: {prompt}")
    response = model(prompt, max_new_tokens=100, temperature=0.6, top_p=0.8, repetition_penalty=2.5, do_sample=True)
    best_response = response[0]['generated_text']
    logging.info(f"Generated text: {best_response}")
    return best_response


def message_and_history(input_text, history, feedback = None):
    """Manage message history and generate responses."""
    if history is None:
        history = []
    output = generate_text(input_text)
    history.append(("User", input_text))
    history.append(("Assistant", output))
    return history, history

def process_feedback(feedback):
    """Log feedback received from the user."""
    logging.info(f"Received feedback: {feedback}")

# Set up the Gradio interface
def setup_interface():
    with gr.Blocks() as block:
        gr.Markdown("<h1><center>Interactive Chatbot</center></h1>")
        chatbot = gr.Chatbot()
        message = gr.Textbox(placeholder="Type your query...")
        feedback = gr.Radio(choices=["Good", "Bad"], label="Feedback on last response")
        state = gr.State()
        submit = gr.Button("Send")
        submit.click(
            fn=message_and_history,
            inputs=[message, state, feedback],
            outputs=[chatbot, state]
        )
    return block

app = setup_interface()
app.launch(debug=True)

Assignments 1 day 2 codes
import gradio as gr

def greet(name):
    return "Hello, " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# To create a shareable link (valid for 72 hours)
demo.launch(share=True)

#########

!pip install huggingface_hub

from huggingface_hub import whoami
from google.colab import userdata

# Get your Hugging Face token from Colab Secrets
hf_token = userdata.get('hf_token')

# Verify the token by checking your identity
try:
    user_info = whoami(token=hf_token)
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Could not log in: {e}")
    print("Please make sure you have added your Hugging Face token to Colab Secrets with the name 'HF_TOKEN'")

########

from datasets import load_dataset

# Load a dataset (e.g., the SQuAD dataset for question answering)
dataset = load_dataset("squad")

# Print information about the dataset
print(dataset)

# Access an example from the training set
print("\nExample from the training set:")
print(dataset["train"][0])

#######

from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("text-generation",
    model="google/flan-t5-base")

# Text to summarize
text = """
Hugging Face is a company and open-source platform that provides tools and models for natural language processing (NLP). It has become a central hub for the ML community, offering a wide range of pre-trained models that can be easily used or fine-tuned for specific applications. Key aspects of Hugging Face include the Transformers library, Model Hub, Datasets library, and Tokenizers library. Hugging Face democratizes access to powerful ML models, making it easier for developers and researchers to build and deploy applications.
"""

# Summarize the text
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

print("Original Text:")
print("text")
print("\nSummary:")
print(summary[0]["generated_text"])






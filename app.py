
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login


model_name_tinyllama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer_tinyllama = AutoTokenizer.from_pretrained(model_name_tinyllama)
model_tinyllama = AutoModelForCausalLM.from_pretrained(model_name_tinyllama,torch_dtype=torch.float32,device_map={"": "cpu"})

def summarize_tinyllama(article):
    # For causal models like TinyLlama, summarization isn't a direct task like with encoder-decoder models.
    # We can prompt it to continue a summary.
    prompt="Summarize the following article clearly and concisely:"
    input_text = f"{prompt}\n{article}\nSummary:"
    inputs = tokenizer_tinyllama(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate tokens - the model will try to complete the input prompt.
    # We need to adjust generation parameters for open-ended generation.
    # max_new_tokens controls how much new text is generated after the prompt.
    outputs = model_tinyllama.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=500,  # Generate up to 300 new tokens for the summary
        do_sample=True,     # Don't sample, use greedy decoding
        temperature=0.7,
        min_new_tokens=150,
        top_p=0.9,
        pad_token_id=tokenizer_tinyllama.eos_token_id, # Pad with EOS token if needed
    )

    # Decode the entire output sequence.
    generated_text = tokenizer_tinyllama.decode(outputs[0], skip_special_tokens=True)


    # The generated text will include the original prompt. We need to extract the summary part.
    # This is a simple approach, more sophisticated parsing might be needed depending on prompt and output.
    summary_start_index = generated_text.find("Summary:") + len("Summary:")
    summary = generated_text[summary_start_index:].strip()

    return summary

def answer_question_tinyllama(article, question):
    # Formulate the prompt to guide the TinyLlama model to answer the question based on the article.
    # We ask the model to act as an AI answering a question based on the provided text.
    input_text = f"From this Article: {article}\n\n Answer the below Question: {question}\n\nAnswer:"

    # Tokenize the input text
    # Truncate if the combined article and question is too long
    inputs = tokenizer_tinyllama(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the answer using the model.
    # We use generate with parameters suitable for generating a concise answer.
    outputs = model_tinyllama.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=500,  # Generate up to 100 new tokens for the answer
        do_sample=True,      # Use sampling to potentially get more varied answers
        temperature=0.7,     # Control randomness
        top_p=0.9,           # Nucleus sampling
        pad_token_id=tokenizer_tinyllama.eos_token_id, # Pad with EOS token if needed
    )
    # Decode the generated sequence
    generated_text = tokenizer_tinyllama.decode(outputs[0], skip_special_tokens=True)

    # The generated text will include the original prompt. We need to extract the answer part.
    # This is a simple approach, more sophisticated parsing might be needed depending on prompt and output.
    answer_start_index = generated_text.find("Answer:") + len("Answer:")
    answer = generated_text[answer_start_index:].strip()

    # Basic cleanup: remove potential repetition of the question or prompt in the answer
    if answer.startswith(question):
        answer = answer[len(question):].strip()

    return answer


st.title("Smart Article Insights Generator")
st.markdown("Summarize an article or ask a question about it.")

mode = st.radio("Select Mode", ["Summarize", "Answer Question"])

article_input = st.text_area("Article Text", height=300, placeholder="Paste the article here...")

question_input = None
if mode == "Answer Question":
    question_input = st.text_input("Question", placeholder="Enter your question here...")

if st.button("Process"):
    if mode == "Summarize":
        if article_input:
            with st.spinner("Generating summary..."):
                output = summarize_tinyllama(article_input)
                st.subheader("Summary")
                st.write(output)
        else:
            st.warning("Please provide an article to summarize.")
    elif mode == "Answer Question":
        if article_input and question_input:
            with st.spinner("Generating answer..."):
                output = answer_question_tinyllama(article_input, question_input)
                st.subheader("Answer")
                st.write(output)
        elif not article_input:
            st.warning("Please provide an article to answer the question from.")
        elif not question_input:
            st.warning("Please provide a question to answer.")

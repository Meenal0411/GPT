import streamlit as st
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
from PIL import Image
import time
import os
import replicate

# Set the option to suppress the warning related to caching
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
@st.cache_data(show_spinner=False)
def generate_llama2_response(prompt_input, string_dialogue, llm, temperature, top_p, max_length, messages):
    start_time = time.time()
    for dict_message in messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm,
                            input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                   "temperature": temperature, "top_p": top_p, "max_length": max_length,
                                   "repetition_penalty": 1})
    end_time = time.time()
    execution_time = end_time - start_time
    if execution_time > 1.0:  # Display only if execution time exceeds 1 second
        st.write(f"generate_llama2_response execution time: {execution_time:.4f} seconds")
    return list(output)

# Load GPT-2 model and tokenizer
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# Function to generate GPT-2 response with caching
@st.cache_data(show_spinner=False)
def generate_gpt2_response(prompt_input):
    input_ids = gpt2_tokenizer.encode(prompt_input, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95,
                                 temperature=0.7)
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    # Set Replicate API token
    #replicate_api = "r8_UT1Zofte3rCH5CzEYooyEtMgUxZKDbY1lLn3P"
    replicate_api = "r8_0QpEN4c6M8jTFIT7cWWru7FpbpfeKvx0PaWaC"
    os.environ["REPLICATE_API_TOKEN"] = replicate_api

    # App title
    st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Chatbot Menu')
        st.write('Choose a chatbot to interact with.')

        option = st.selectbox("Select a Chatbot", ["GPT-2", "Llama 2", "About Us"])

        if option == "Llama 2":
            st.subheader('Models and parameters')
            selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-13B'], key='selected_model')
            if selected_model == 'Llama2-13B':
                llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
            temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
            top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
            max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    if st.sidebar.button('Clear Chat History', on_click=clear_chat_history):
        return  # This should be indented inside the if statement

    # User-provided prompt
    if option == "Llama 2":
        if prompt := st.chat_input(disabled=not replicate_api):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Generate a new response if the last message is not from the assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_llama2_response(prompt,
                                                             "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.",
                                                             llm, temperature, top_p, max_length,
                                                             st.session_state.messages)
                        placeholder = st.empty()
                        full_response = ''
                        for item in response:
                            full_response += item
                            placeholder.markdown(full_response)
                        placeholder.markdown(full_response)

    elif option == "GPT-2":
        st.markdown("## Welcome to the GPT-2 Chatbot")

        user_input = st.text_input("You:", "")

        if st.button("Send"):
            if not user_input:
                st.warning("Please enter input.")
            else:
                user_input = user_input.strip()

                if not any(c.isalpha() for c in user_input):
                    st.warning("Invalid input. Please enter a meaningful input.")
                else:
                    response = generate_gpt2_response(user_input)
                    st.text("Bot: {}".format(response))

    elif option == "About Us":
        st.markdown("## About Us")
        st.write("This chatbot is powered by Hugging Face's transformers library. It uses a combination of a Question-Answering model (distilbert-base-cased-distilled-squad) for specific questions and a GPT-2 language model for general conversation. The second chatbot is the Llama 2 chatbot.")
        st.write("Feel free to interact with the chatbot on the 'Home' page!")

        # Add another image in the About Us page
        #about_us_image = Image.open("C:/Users/Sachin/OneDrive/Documents/New/chatbot.png")
        #st.image(about_us_image, caption="Another Image", use_column_width=True)

if __name__ == "__main__":
    main()

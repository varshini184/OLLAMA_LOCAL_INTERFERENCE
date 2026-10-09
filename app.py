
import streamlit as st
import ollama

st.set_page_config(
    page_title="Ollama Local Interface",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Ollama Local Inference")
st.write("Ask a question and get a response from your local Llama 3.2 model.")

model = "llama3.2"

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Example: Explain artificial intelligence in simple words.",
    height=150
)

if st.button("Generate Response", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Generating response..."):
            try:
                result = ollama.chat(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt.strip()
                        }
                    ]
                )

                answer = result["message"]["content"]

                st.subheader("Model Response")
                st.markdown(answer)

            except Exception as error:
                st.error(f"Could not generate a response: {error}")
                st.info(
                    "Check that Ollama is running and "
                    "the llama3.2 model is installed."
                )

st.divider()
st.caption("Powered by Ollama and Llama 3.2 · Runs locally")
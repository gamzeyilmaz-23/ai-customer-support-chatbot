import streamlit as st
from chatbot import SupportChatbot

st.set_page_config(page_title="AI Customer Support Chatbot", page_icon="💬", layout="centered")

@st.cache_resource
def load_chatbot():
    return SupportChatbot("data/knowledge_base.json")

bot = load_chatbot()

st.title("💬 Customer Support Chatbot")
st.caption("A Python NLP portfolio demo using TF-IDF and cosine similarity.")
st.info(
    "Demo project: the FAQ content is sample information, not official FOREO product guidance. "
    "For product-specific instructions, always consult the manufacturer's official manual or support team."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm the demo support assistant. What can I help you with?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a support question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer, score, matched_question = bot.get_answer(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
        if matched_question:
            st.caption(f"Matched FAQ: {matched_question} · Similarity: {score:.2f}")
    st.session_state.messages.append({"role": "assistant", "content": answer})

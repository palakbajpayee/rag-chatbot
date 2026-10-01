"""Web chat UI:  streamlit run app.py"""
import anthropic
import streamlit as st

import config
from chatbot import SupportBot

st.set_page_config(page_title=f"{config.STORE_NAME} Support", page_icon="🛍️")
st.title(f"{config.STORE_NAME} Customer Support")

if "bot" not in st.session_state:
    st.session_state.bot = SupportBot()
    st.session_state.history = [("assistant", "Hi! I can help with orders, shipping, returns, "
                                               "refunds, payments and products. What can I do for you?")]

with st.sidebar:
    st.caption(f"Model: `{config.MODEL}` · effort: `{config.EFFORT}`")
    show_tools = st.toggle("Show tool calls", value=False)
    if st.button("New conversation"):
        st.session_state.bot.reset()
        st.session_state.history = st.session_state.history[:1]
        st.rerun()

for role, text in st.session_state.history:
    st.chat_message(role).markdown(text)

if prompt := st.chat_input("Type your question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.history.append(("user", prompt))
    with st.chat_message("assistant"):
        with st.spinner("Looking that up..."):
            try:
                reply = st.session_state.bot.chat(prompt)
                text = reply.text
                if show_tools and reply.tool_calls:
                    with st.expander("Tool calls"):
                        st.json(reply.tool_calls)
            except anthropic.AuthenticationError:
                text = "Missing or invalid `ANTHROPIC_API_KEY`."
            except anthropic.RateLimitError:
                text = "We're busy right now - please try again in a moment."
            except (anthropic.APIStatusError, anthropic.APIConnectionError):
                text = "Something went wrong reaching the assistant. Please try again."
        st.markdown(text)
    st.session_state.history.append(("assistant", text))

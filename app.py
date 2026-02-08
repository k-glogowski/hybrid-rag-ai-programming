"""
Proste GUI w Streamlit – czat wywołujący workflow z orchestratora.
Użytkownik podaje temat dokumentacji, a aplikacja generuje krótką dokumentację (RAG).
"""

import streamlit as st
from orchestrator import run_orchestrator

st.set_page_config(
    page_title="Generator dokumentacji Docker",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Generator dokumentacji Docker")
st.caption("Podaj temat – wygeneruję krótką dokumentację na podstawie bazy wiedzy Docker.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Wpisz temat dokumentacji (np. Instalacja Docker Desktop na Linuxie)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generuję dokumentację..."):
            try:
                report = run_orchestrator(prompt)
                st.markdown(report)
                st.session_state.messages.append({"role": "assistant", "content": report})
            except Exception as e:
                error_msg = f"Błąd: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

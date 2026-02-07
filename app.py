"""
Streamlit chat GUI do interakcji z workflow RAG (Docker docs).
"""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from workflow import create_graph, AVAILABLE_MODELS, DEFAULT_RESPONSE_MODEL, DEFAULT_GRADER_MODEL


def main():
    st.set_page_config(
        page_title="Chat – Dokumentacja Docker",
        page_icon="🐳",
        layout="wide",
    )

    # Toolbox – panel po lewej
    toolbox_col, main_col = st.columns([1, 4])

    with toolbox_col:
        st.subheader("⚙️ Toolbox")
        response_model = st.selectbox(
            "Model (response)",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_RESPONSE_MODEL) if DEFAULT_RESPONSE_MODEL in AVAILABLE_MODELS else 0,
            key="response_model",
        )
        grader_model = st.selectbox(
            "Model (grader)",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_GRADER_MODEL) if DEFAULT_GRADER_MODEL in AVAILABLE_MODELS else 0,
            key="grader_model",
        )

    with main_col:
        st.title("🐳 Chat – Dokumentacja Docker")
        st.caption("Zadawaj pytania o Docker. Workflow używa RAG, aby odpowiadać na podstawie dokumentacji.")

        # Inicjalizacja historii czatu w session_state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Wyświetl historię czatu
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Pole wejściowe
        if prompt := st.chat_input("Napisz pytanie o Docker..."):
            # Dodaj wiadomość użytkownika
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Wyświetl wiadomość użytkownika
            with st.chat_message("user"):
                st.markdown(prompt)

            # Przygotuj listę wiadomości dla LangGraph (MessagesState)
            langchain_messages = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    langchain_messages.append(HumanMessage(content=m["content"]))
                else:
                    langchain_messages.append(AIMessage(content=m["content"]))

            # Wywołaj workflow z wybranymi modelami
            with st.chat_message("assistant"):
                with st.spinner("Przetwarzam zapytanie..."):
                    try:
                        graph = create_graph(
                            response_model_name=response_model,
                            grader_model_name=grader_model,
                        )
                        result = graph.invoke({"messages": langchain_messages})
                        # Pobierz ostatnią odpowiedź AI (AIMessage)
                        response = None
                        for msg in reversed(result["messages"]):
                            if isinstance(msg, AIMessage) and msg.content:
                                response = msg.content if isinstance(msg.content, str) else str(msg.content)
                                break
                        if response is None:
                            response = "Nie udało się wygenerować odpowiedzi."
                    except Exception as e:
                        response = f"❌ Błąd: {str(e)}"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

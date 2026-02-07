"""
Streamlit GUI do chatu z workflow RAG (Docker docs).
Technical details (pełny stan flow) w accordion "details".
"""
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
from langchain_core.messages import HumanMessage

from build_index import delete_index, rebuild_index
from workflow import get_graph, clear_graph_cache


def message_to_dict(msg):
    """Konwersja wiadomości LangChain do słownika do wyświetlenia."""
    d = {
        "type": type(msg).__name__,
        "content": getattr(msg, "content", None) or "",
    }
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
        d["additional_kwargs"] = msg.additional_kwargs
    return d


def state_to_display(state):
    """Konwersja stanu workflow do czytelnej reprezentacji (historia flow)."""
    if not state or "messages" not in state:
        return {}
    return {
        "messages": [message_to_dict(m) for m in state["messages"]],
    }


st.set_page_config(page_title="Chat – Docker docs RAG", page_icon="🐳", layout="wide")
st.title("🐳 Chat z dokumentacją Docker")
st.caption("Zadaj pytanie – workflow może wyszukać fragmenty dokumentacji i odpowiedzieć.")

# Toolbox (sidebar) – ustawienia retrievera i indeksu Chroma
with st.sidebar:
    st.header("🔧 Toolbox")

    st.subheader("Wyszukiwanie")
    retriever_k = st.slider(
        "Liczba chunków z retrievera",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="Ile fragmentów dokumentacji zwraca wyszukiwarka przy jednym zapytaniu.",
    )

    st.subheader("Indeks Chroma (chunki)")
    chunk_size = st.number_input(
        "Długość chunka",
        min_value=100,
        max_value=2000,
        value=400,
        step=50,
        help="Rozmiar fragmentu tekstu przy dzieleniu dokumentów (znaki/tokeny).",
    )
    chunk_overlap = st.number_input(
        "Nakładka chunków (overlap)",
        min_value=0,
        max_value=500,
        value=100,
        step=10,
        help="Ile znaków wspólnych między sąsiednimi chunkami.",
    )

    col_del, col_rebuild = st.columns(2)
    with col_del:
        if st.button("🗑️ Usuń indeks", help="Czyści dane indeksu z bazy Chroma (bez usuwania plików)."):
            clear_graph_cache()  # zwolnij referencje do Chroma przed czyszczeniem
            removed = delete_index()
            if removed:
                st.success("Indeks usunięty.")
            else:
                st.info("Brak indeksu do usunięcia.")
            st.rerun()
    with col_rebuild:
        if st.button("🔄 Przebuduj indeks", help="Czyści dane indeksu i buduje od zera z powyższymi parametrami chunków (bez usuwania plików)."):
            with st.spinner("Przebudowuję indeks..."):
                clear_graph_cache()  # zwolnij referencje do Chroma przed czyszczeniem
                rebuild_index(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.success("Indeks przebudowany.")
            st.rerun()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Details (stan flow) przy każdej wiadomości asystenta
        if msg["role"] == "assistant" and msg.get("flow_state") is not None:
            with st.expander("details", expanded=False):
                st.caption("Technical details – stan flow dla tej odpowiedzi")
                display_state = state_to_display(msg["flow_state"])
                st.json(display_state)

if prompt := st.chat_input("Twoje pytanie..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Przetwarzam..."):
            graph = get_graph(retriever_k=retriever_k)
            messages_for_graph = [HumanMessage(content=prompt)]
            result = graph.invoke({"messages": messages_for_graph})

        answer = result["messages"][-1].content
        st.markdown(answer)
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
            "flow_state": result,
        })
        # Wyświetl details od razu dla nowej odpowiedzi
        with st.expander("details", expanded=False):
            st.caption("Technical details – stan flow dla tej odpowiedzi")
            display_state = state_to_display(result)
            st.json(display_state)

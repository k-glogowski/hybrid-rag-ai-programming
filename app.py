"""
Streamlit GUI: chat z retrieverem, ustawienia chunków, czyszczenie bazy, ładowanie danych.
"""
import streamlit as st
from build_index import (
    get_retriever,
    clear_chroma_db,
    load_data_to_vectorstore,
)

st.set_page_config(
    page_title="RAG Retriever — Docker Docs",
    page_icon="🔍",
    layout="wide",
)

# Sidebar: ustawienia
with st.sidebar:
    st.header("⚙️ Ustawienia")

    k = st.slider(
        "Liczba zwracanych chunków (k)",
        min_value=1,
        max_value=20,
        value=4,
        help="Ile fragmentów dokumentacji zwracać przy każdym zapytaniu.",
    )

    st.divider()
    st.subheader("Indeks wektorowy")

    chunk_size = st.number_input(
        "Chunk size",
        min_value=100,
        max_value=2000,
        value=400,
        step=50,
        help="Rozmiar fragmentu w tokenach (tiktoken).",
    )
    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=10,
        help="Nakładka między fragmentami (tokeny).",
    )

    if st.button("🗑️ Wyczyść bazę danych", type="secondary"):
        if clear_chroma_db():
            st.success("Baza Chroma usunięta.")
            st.rerun()
        else:
            st.info("Brak bazy do usunięcia.")

    if st.button("📥 Załaduj dane do bazy", type="primary"):
        with st.spinner("Pobieranie danych, dzielenie i budowanie indeksu..."):
            n = load_data_to_vectorstore(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
            )
        if n is not None:
            st.success(f"Zapisano {n:,} chunków.")
            st.rerun()
        else:
            st.error("Nie udało się wczytać danych (brak parquet lub błąd).")

# Główny obszar: chat
st.title("🔍 Chat z retrieverem — Docker Docs")
st.caption("Zadaj pytanie; retriever zwróci najbardziej pasujące fragmenty dokumentacji.")

# Stan sesji: historia wiadomości
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetl historię
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander(f"📄 Zwrócone chunki ({len(msg['chunks'])})"):
                for i, doc in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i}** (score/metadata w razie dostępności)")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                    if doc.metadata:
                        st.caption(str(doc.metadata))

# Retriever z aktualnym k (odświeżany przy każdej interakcji)
retriever = get_retriever(k=k)

if retriever is None:
    st.warning(
        "Baza wektorowa jest pusta. Użyj **Załaduj dane do bazy** w panelu bocznym "
        "(ustaw chunk size i overlap), aby zbudować indeks."
    )

# Pole wpisu i wysłanie
if prompt := st.chat_input("Zadaj pytanie o dokumentację Docker..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if retriever is None:
            reply = "Nie mogę wyszukać — najpierw załaduj dane do bazy (sidebar)."
            st.markdown(reply)
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "chunks": [],
            })
        else:
            try:
                docs = retriever.invoke(prompt)
                if not docs:
                    reply = "Brak pasujących fragmentów."
                    st.markdown(reply)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "chunks": [],
                    })
                else:
                    reply = f"Znaleziono **{len(docs)}** fragmentów (k={k})."
                    st.markdown(reply)
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"Chunk {i}"):
                            st.write(doc.page_content)
                            if doc.metadata:
                                st.caption(doc.metadata)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "chunks": docs,
                    })
            except Exception as e:
                err = f"Błąd retrievera: {e}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err,
                    "chunks": [],
                })

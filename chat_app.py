"""
Proste GUI (Streamlit) do chatu z workflow_summary – RAG dokumentacji Docker.
"""
import streamlit as st
from langchain_core.messages import HumanMessage

# Import workflow – załaduj przed użyciem
from workflow_summary import graph, make_config, get_thread_state

st.set_page_config(page_title="Chat – Dokumentacja Docker", page_icon="🐳", layout="centered")

st.title("🐳 Chat z dokumentacją Docker")
st.caption("Zadawaj pytania o Docker – asystent szuka w dokumentacji i odpowiada.")

# Inicjalizacja stanu sesji
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-chat-1"
if "messages" not in st.session_state:
    st.session_state.messages = []

def _serialize_state_snapshot(state_snapshot):
    """Serializuje pełny StateSnapshot grafu (values, next, metadata, created_at)."""
    vals = state_snapshot.values or {}
    msgs = vals.get("messages", [])
    serialized_msgs = []
    for m in msgs:
        kind = m.__class__.__name__
        content = getattr(m, "content", str(m))
        serialized_msgs.append({"type": kind, "content": content if isinstance(content, str) else str(content)})
    meta = getattr(state_snapshot, "metadata", None)
    try:
        meta_serial = dict(meta) if meta else None
    except Exception:
        meta_serial = str(meta) if meta else None
    return {
        "values": {"messages": serialized_msgs, "summary": vals.get("summary") or ""},
        "next": tuple(getattr(state_snapshot, "next", ()) or ()),
        "metadata": meta_serial,
        "created_at": getattr(state_snapshot, "created_at", None),
        "config": {"configurable": dict((getattr(state_snapshot, "config", None) or {}).get("configurable") or {})},
    }


def _render_state(snapshot_data):
    """Wyświetla dokładny aktualny stan grafu na chwilę odpowiedzi."""
    with st.expander("📋 Stan grafu (StateSnapshot)", expanded=False):
        vals = snapshot_data.get("values", {})
        st.write("**values.summary:**")
        st.write(vals.get("summary") or "(brak)")
        st.write("**values.messages:**")
        for i, m in enumerate(vals.get("messages", [])):
            st.write(f"*{i+1}. {m['type']}*:")
            st.text(m["content"])
        st.write("**next:**", snapshot_data.get("next", ()))
        if snapshot_data.get("config"):
            st.write("**config.configurable:**", snapshot_data["config"].get("configurable", {}))
        if snapshot_data.get("created_at"):
            st.write("**created_at:**", snapshot_data["created_at"])
        if snapshot_data.get("metadata") is not None:
            st.write("**metadata:**")
            meta = snapshot_data["metadata"]
            if isinstance(meta, dict):
                st.json(meta)
            else:
                st.text(str(meta))


# Wyświetl historię czatu
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, dict) and msg.get("role") == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and (state_snapshot := msg.get("state")):
            _render_state(state_snapshot)

# Wejście użytkownika
if prompt := st.chat_input("Zadaj pytanie o Docker..."):
    # Dodaj wiadomość użytkownika
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Wywołaj workflow
    with st.chat_message("assistant"):
        with st.spinner("Szukam w dokumentacji..."):
            try:
                config = make_config(st.session_state.thread_id)
                print(config)

                # Przekaż summary z poprzedniego stanu (wieloturowość)
                prev_state = graph.get_state(config)
                inp = {"messages": [HumanMessage(content=prompt)]}
                if prev_state.values.get("summary"):
                    inp["summary"] = prev_state.values["summary"]

                result = graph.invoke(inp, config=config)

                # Ostatnia wiadomość AI to odpowiedź
                last_msg = result["messages"][-1]
                answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

            except Exception as e:
                answer = f"❌ Błąd: {e}"

        st.markdown(answer)

        # Dokładny aktualny StateSnapshot grafu na chwilę odpowiedzi
        state_snapshot = None
        try:
            snapshot = get_thread_state(st.session_state.thread_id)
            state_snapshot = _serialize_state_snapshot(snapshot)
            _render_state(state_snapshot)
        except Exception:
            pass

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "state": state_snapshot,
    })

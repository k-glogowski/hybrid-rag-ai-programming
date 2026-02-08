import os
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from retriever import get_retriever
from langchain_core.tools import create_retriever_tool
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.prebuilt import ToolNode, tools_condition


def last_user_content(messages):
    """Treść ostatniego pytania użytkownika w wątku (dla wieloturowej rozmowy)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return (messages[0].content if messages else "") if messages else ""


class State(MessagesState):
    summary: str


def mark_old_messages_removed(state: State):
    """
    Oznacza wszystkie wiadomości oprócz ostatniej jako Removed (do usunięcia przy merge),
    ale tylko gdy summary jest zdefiniowane i nie jest pustym stringiem.
    """
    summary = (state.get("summary") or "").strip()
    if not summary:
        return {}
    messages = state.get("messages") or []
    if len(messages) <= 1:
        return {}
    to_remove = []
    for m in messages[:-1]:
        msg_id = getattr(m, "id", None)
        if msg_id is not None:
            to_remove.append(RemoveMessage(id=msg_id))
    if not to_remove:
        return {}
    return {"messages": to_remove}


SUMMARY_PROMPT = (
    "Podsumuj krótko (2–4 zdania) całą rozmowę: pytania użytkownika i udzielone odpowiedzi. "
    "Skup się na tematach i wnioskach. Odpowiadaj po polsku."
)

response_model = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

grader_model = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

def summarize_conversation(state: State):
    messages = state["messages"] + [
        HumanMessage(content=SUMMARY_PROMPT),
    ]
    response = response_model.invoke(messages)
    return {"summary": response.content}


retriever = get_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "search_docker_docs",
    "Wyszukuj fragmenty dokumentacji Docker (instrukcje, API, konfiguracja, opisy).",
)

def generate_query_or_respond(state: MessagesState):
    """
    LLM decyduje: wywołać tool search_docker_docs (RAG) albo odpowiedzieć od razu.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

GRADE_PROMPT = (
    "Jesteś graderem oceniającym, czy znaleziony fragment dokumentacji Docker jest istotny względem pytania użytkownika.\n"
    "Oto fragment dokumentacji:\n\n{context}\n\n"
    "Oto pytanie użytkownika:\n{question}\n\n"
    "Jeśli fragment pasuje do intencji pytania (temat, komendy, konfiguracja), odpowiedz 'yes'. "
    "Jeśli nie pasuje – 'no'."
)

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' jeśli istotne, 'no' jeśli nie"
    )

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Sprawdza, czy zwrócone fragmenty dokumentacji są istotne."""
    question = last_user_content(state["messages"])
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score.strip().lower()
    return "generate_answer" if score == "yes" else "rewrite_question"

REWRITE_PROMPT = (
    "Popraw pytanie użytkownika tak, aby było bardziej precyzyjne w kontekście dokumentacji Docker.\n"
    "Weź pod uwagę: komendy, konfigurację, API, konkretne funkcje lub narzędzia.\n"
    "Oto pytanie:\n-------\n{question}\n-------\n"
    "Zwróć jedno, lepsze pytanie."
)

def rewrite_question(state: MessagesState):
    question = last_user_content(state["messages"])
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


# --- Generate answer ---
GENERATE_PROMPT = (
    "Jesteś asystentem pomagającym w dokumentacji Docker.\n"
    "Korzystaj z poniższego kontekstu (fragmenty dokumentacji), aby odpowiedzieć na pytanie.\n"
    "Jeśli nie wiesz – napisz, że na podstawie dostępnej dokumentacji nie możesz odpowiedzieć.\n"
    "Maksymalnie 4 zdania. Odpowiadaj po polsku.\n\n"
    "{summary_block}"
    "Pytanie: {question}\n\nKontekst z dokumentacji:\n{context}"
)


def generate_answer(state: State):
    question = last_user_content(state["messages"])
    context = state["messages"][-1].content
    summary = (state.get("summary") or "").strip()
    summary_block = (
        f"Poprzednie podsumowanie rozmowy:\n{summary}\n\n"
        if summary else ""
    )
    prompt = GENERATE_PROMPT.format(
        question=question,
        context=context,
        summary_block=summary_block,
    )
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


workflow = StateGraph(State)
workflow.add_node("startowy", mark_old_messages_removed)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "startowy")
workflow.add_edge("startowy", "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: "summarize_conversation"},
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", "summarize_conversation")
workflow.add_edge("summarize_conversation", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Checkpointer zapisuje stan rozmowy per thread_id – możesz odczytać stan i historię
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def make_config(thread_id: str):
    """Konfiguracja z thread_id – ten sam thread_id = ta sama rozmowa (checkpointy)."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


def get_thread_state(thread_id: str):
    """Podgląd aktualnego stanu wątku (ostatni checkpoint)."""
    config = make_config(thread_id)
    state = graph.get_state(config)
    return state


def get_thread_history(thread_id: str, limit: int = 20):
    """Historia checkpointów wątku – przegląd stanów w czasie."""
    config = make_config(thread_id)
    history = graph.get_state_history(config, limit=limit)
    return list(history)


# Usunięcie threadu: MemorySaver trzyma dane w pamięci; żeby „usunąć” wątek,
# użyj nowego thread_id. Dla trwałego usuwania (np. SQLite/Postgres) trzeba
# checkpointer z API delete (np. SqliteSaver).

if __name__ == "__main__":
    thread_id = "demo-docker-1"
    config = make_config(thread_id)

    # Jedno wywołanie – stan zapisany w checkpointerze pod thread_id
    result = graph.invoke(
        {"messages": [HumanMessage(content="Jak zainstalować Docker Desktop na Linuxie?")]},
        config=config,
    )
    print("Odpowiedź:", result["messages"][-1].content)
    print("Podsumowanie:", result.get("summary", "(brak)"))

    # Drugi invoke: jawnie przekazujemy stan z checkpointu (w tym streszczenie),
    # żeby merge był czytelny; checkpointer i tak załaduje ten stan, ale tak widać,
    # że summary z poprzedniego runu jest dostępne.
    prev_state = graph.get_state(config)
    second_input = {
        "messages": [HumanMessage(content="Oj to smutmo mi tego nie umiem zainstalować")],
    }
    # "messages": [HumanMessage(content="Na pewno tak? z tymi instrukcjami?")],

    if prev_state.values.get("summary"):
        second_input["summary"] = prev_state.values["summary"]
    result = graph.invoke(second_input, config=config)
    print("Odpowiedź:", result["messages"][-1].content)
    print("Podsumowanie:", result.get("summary", "(brak)"))

    # Podgląd stanu wątku (to samo co result, jeśli nie było dalszych wywołań)
    current = get_thread_state(thread_id)
    print("\n--- Stan wątku (get_state) ---")
    print("Wiadomości:", len(current.values.get("messages", [])))
    print("Summary:", current.values.get("summary", "(brak)"))

    # Historia checkpointów (każdy krok grafu może dodać checkpoint)
    history = get_thread_history(thread_id)
    print("\n--- Historia wątku (get_state_history) ---")
    for i, state in enumerate(history):
        msg_count = len(state.values.get("messages", []))
        s = (state.values.get("summary") or "")[:50]
        print(f"  checkpoint {i}: {msg_count} wiadomości, summary={s!r}...")

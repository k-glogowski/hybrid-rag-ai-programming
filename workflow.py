import logging
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.chat_models import init_chat_model
from retriever import get_retriever

logger = logging.getLogger(__name__)
from langchain_core.tools import create_retriever_tool
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition

response_model = init_chat_model(model="gpt-4o", temperature=0.0)
grader_model = init_chat_model(model="gpt-4o-mini", temperature=0.0)


def _build_graph(retriever_k: int = 4):
    """Buduje skompilowany graf z retrieverem zwracającym retriever_k chunków."""
    retriever = get_retriever(k=retriever_k)
    retriever_tool = create_retriever_tool(
        retriever,
        "search_docker_docs",
        "Wyszukuj fragmenty dokumentacji Docker (instrukcje, API, konfiguracja, opisy).",
    )
    return _compile_workflow(retriever_tool)

def _make_generate_query_or_respond(retriever_tool):
    def generate_query_or_respond(state: MessagesState):
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}
    return generate_query_or_respond

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
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score.strip().lower()
    if score != "yes":
        logger.info("Grader: kontekst nieistotny -> ścieżka rewrite_question")
    return "generate_answer" if score == "yes" else "rewrite_question"

REWRITE_PROMPT = (
    "Popraw pytanie użytkownika tak, aby było bardziej precyzyjne w kontekście dokumentacji Docker.\n"
    "Weź pod uwagę: komendy, konfigurację, API, konkretne funkcje lub narzędzia.\n"
    "Oto pytanie:\n-------\n{question}\n-------\n"
    "Zwróć jedno, lepsze pytanie."
)

def rewrite_question(state: MessagesState):
    logger.info("Użyto metody rewrite_question (grader ocenił kontekst jako nieistotny)")
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


# --- Generate answer ---
GENERATE_PROMPT = (
    "Jesteś asystentem pomagającym w dokumentacji Docker.\n"
    "Korzystaj z poniższego kontekstu (fragmenty dokumentacji), aby odpowiedzieć na pytanie.\n"
    "Jeśli nie wiesz – napisz, że na podstawie dostępnej dokumentacji nie możesz odpowiedzieć.\n"
    "Maksymalnie 4 zdania. Odpowiadaj po polsku.\n\n"
    "Pytanie: {question}\n\nKontekst z dokumentacji:\n{context}"
)


def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


def _compile_workflow(retriever_tool):
    workflow = StateGraph(MessagesState)
    workflow.add_node("generate_query_or_respond", _make_generate_query_or_respond(retriever_tool))
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


_graph_cache: dict[int, object] = {}


def get_graph(retriever_k: int = 4):
    """Zwraca skompilowany graf. retriever_k – liczba chunków zwracanych przez retriever. Wynik jest cache’owany per k."""
    if retriever_k not in _graph_cache:
        _graph_cache[retriever_k] = _build_graph(retriever_k)
    return _graph_cache[retriever_k]


def clear_graph_cache():
    """Czyści cache grafów (np. po przebudowie indeksu Chroma)."""
    _graph_cache.clear()


if __name__ == "__main__":
    graph = get_graph(retriever_k=4)
    result = graph.invoke({"messages": [HumanMessage(content="Jak zainstalować Docker Desktop na Linuxie?")]})
    print(result["messages"][-1].content)
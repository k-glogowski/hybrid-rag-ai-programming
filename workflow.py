import os
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from retriever import get_retriever
from langchain_core.tools import create_retriever_tool
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition

DEFAULT_RESPONSE_MODEL = "openai/gpt-4o"
DEFAULT_GRADER_MODEL = "deepseek/deepseek-v3.2-speciale"

AVAILABLE_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-70b-instruct",
    "deepseek/deepseek-v3.2-speciale",
]


def _create_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0.0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )


retriever = get_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "search_docker_docs",
    "Wyszukuj fragmenty dokumentacji Docker (instrukcje, API, konfiguracja, opisy).",
)

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

REWRITE_PROMPT = (
    "Popraw pytanie użytkownika tak, aby było bardziej precyzyjne w kontekście dokumentacji Docker.\n"
    "Weź pod uwagę: komendy, konfigurację, API, konkretne funkcje lub narzędzia.\n"
    "Oto pytanie:\n-------\n{question}\n-------\n"
    "Zwróć jedno, lepsze pytanie."
)

GENERATE_PROMPT = (
    "Jesteś asystentem pomagającym w dokumentacji Docker.\n"
    "Korzystaj z poniższego kontekstu (fragmenty dokumentacji), aby odpowiedzieć na pytanie.\n"
    "Jeśli nie wiesz – napisz, że na podstawie dostępnej dokumentacji nie możesz odpowiedzieć.\n"
    "Maksymalnie 4 zdania. Odpowiadaj po polsku.\n\n"
    "Pytanie: {question}\n\nKontekst z dokumentacji:\n{context}"
)


def _get_last_user_question(messages: list) -> str:
    """Pobiera treść ostatniego pytania użytkownika (działa przy jednej wiadomości i przy pełnej historii z app.py)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and getattr(m, "content", None):
            content = m.content
            return content if isinstance(content, str) else str(content)
    if messages:
        first = messages[0]
        content = getattr(first, "content", "")
        return content if isinstance(content, str) else str(content)
    return ""

def _get_context_as_string(messages: list) -> str:
    """Ostatnia wiadomość to wynik retrievera; content może być str lub list."""
    if not messages:
        return ""
    raw = messages[-1].content
    if isinstance(raw, list):
        return "\n\n".join(str(x) for x in raw)
    return str(raw) if raw is not None else ""


def create_graph(
    response_model_name: str = DEFAULT_RESPONSE_MODEL,
    grader_model_name: str = DEFAULT_GRADER_MODEL,
):
    """Tworzy skompilowany graf workflow z wybranymi modelami."""
    response_model = _create_llm(response_model_name)
    grader_model = _create_llm(grader_model_name)

    def generate_query_or_respond(state: MessagesState):
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        question = _get_last_user_question(state["messages"])
        context = _get_context_as_string(state["messages"])
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        score = response.binary_score.strip().lower()
        return "generate_answer" if score == "yes" else "rewrite_question"

    def rewrite_question(state: MessagesState):
        question = _get_last_user_question(state["messages"])
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    def generate_answer(state: MessagesState):
        question = _get_last_user_question(state["messages"])
        context = _get_context_as_string(state["messages"])
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

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


graph = create_graph()

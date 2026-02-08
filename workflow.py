import os
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from retriever import get_retriever
from langchain_core.tools import create_retriever_tool
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition

response_model = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)
grader_model = ChatOpenAI(
    model="deepseek/deepseek-v3.2-speciale",
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
    question = state["messages"][0].content
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


workflow = StateGraph(MessagesState)
workflow.add_node(generate_query_or_respond)
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

graph = workflow.compile()

if __name__ == "__main__":
    result = graph.invoke({"messages": [HumanMessage(content="Jak zainstalować Docker Desktop na Linuxie?")]})
    print(result["messages"][-1].content)

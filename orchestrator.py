from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import operator

import os
from dotenv import load_dotenv
load_dotenv()
from retriever import get_retriever

def retrieve_context(query: str):
    retriever = get_retriever()
    results = retriever.invoke(query)
    return "\n\n".join([r.page_content for r in results])

class Section(BaseModel):
    name: str = Field(description="Tytuł sekcji dokumentacji.")
    description: str = Field(description="Krótki opis, co sekcja ma obejmować.")

class Sections(BaseModel):
    sections: list[Section] = Field(description="Lista sekcji dokumentacji.")

SECTION_GENERATE_PROMPT = (
    "Napisz jedną sekcję dokumentacji Docker na podstawie podanego kontekstu. "
    "Tytuł sekcji: {name}. Opis zakresu: {description}. "
    "Użyj wyłącznie informacji z kontekstu. Bez wstępu, samą treść sekcji. Markdown."
)

response_model = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

planner = response_model.with_structured_output(Sections)

class OrchestratorState(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list[str], operator.add]
    final_report: str

def orchestrator_node(state: OrchestratorState):
    report_sections = planner.invoke(
        [
            SystemMessage(
                content="Zaplanuj sekcje krótkiej dokumentacji (3–5 sekcji). Każda sekcja: name, krótki description."
            ),
            HumanMessage(content=f"Temat dokumentacji: {state['topic']}"),
        ]
    )
    return {"sections": report_sections.sections}

def assign_doc_workers(state: OrchestratorState):
    return [
        Send("doc_worker", {"section": s, "topic": state["topic"]})
        for s in state["sections"]
    ]

def doc_worker(state: dict):
    section = state["section"]
    if isinstance(section, dict):
        name = section.get("name", "")
        desc = section.get("description", "")
    else:
        name = getattr(section, "name", "")
        desc = getattr(section, "description", "")
    topic = state.get("topic", "")
    query = f"{name} {desc}"
    context = retrieve_context(query)
    prompt = SECTION_GENERATE_PROMPT.format(name=name, description=desc)
    prompt = f"{prompt}\n\nKontekst z dokumentacji Docker:\n{context}"
    msg = response_model.invoke([{"role": "user", "content": prompt}])
    return {"completed_sections": [msg.content]}

def doc_synthesizer(state: OrchestratorState):
    report = "\n\n---\n\n".join(state["completed_sections"])
    return {"final_report": report}  


orchestrator_builder = StateGraph(OrchestratorState)
orchestrator_builder.add_node("orchestrator", orchestrator_node)
orchestrator_builder.add_node("doc_worker", doc_worker)
orchestrator_builder.add_node("synthesizer", doc_synthesizer)
orchestrator_builder.add_edge(START, "orchestrator")
orchestrator_builder.add_conditional_edges(
    "orchestrator", assign_doc_workers, ["doc_worker"]
)
orchestrator_builder.add_edge("doc_worker", "synthesizer")
orchestrator_builder.add_edge("synthesizer", END)

orchestrator = orchestrator_builder.compile()

if __name__ == "__main__":
    result = orchestrator.invoke({"topic": "Instalacja Docker Desktop na Linuxie"})
    print(result["final_report"])
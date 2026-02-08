import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain_core.messages import HumanMessage
from workflow import graph

# Model do ewaluacji (LLM-as-judge)
eval_model = ChatOpenAI(
    model="openai/gpt-5.2",
    temperature=0.0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)


class EvalScore(BaseModel):
    score: float = Field(description="Ocena 0.0–1.0: jak dobrze odpowiedź modelu pokrywa oczekiwaną (1.0 = w pełni poprawna).")
    comment: str = Field(default="", description="Krótkie uzasadnienie oceny.")


EVAL_PROMPT = """Oceń, na ile odpowiedź asystenta jest poprawna względem oczekiwanej.

Pytanie użytkownika: {question}

Oczekiwana (wzorcowa) odpowiedź: {expected}

Odpowiedź asystenta: {actual}

Podaj score od 0.0 do 1.0 (1.0 = odpowiedź w pełni poprawna / pokrywa oczekiwaną, 0.0 = całkowicie błędna lub nie na temat)."""

client = Client()
dataset_name = "Docker_RAG_Evaluation_v1"

# Przykładowe dane testowe
examples = [
    ("Czym jest docker-compose.yml?", "To plik konfiguracyjny YAML służący do definiowania i uruchamiania wielokontenerowych aplikacji Docker."),
    ("Jak usunąć wszystkie nieużywane obrazy?", "Użyj komendy docker image prune -a."),
    ("Jak zainstalować Docker Desktop na Ubuntu?", "Należy pobrać najnowszy pakiet .deb i użyć komendy sudo apt-get install ./docker-desktop.deb."),
]


# Pobierz lub utwórz dataset; dodaj tylko nowe przykłady (bez duplikatów po pytaniu)
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name, description="Testy bazowe dla bota dokumentacji Docker.")
    existing_questions = set()
else:
    dataset = client.read_dataset(dataset_name=dataset_name)
    existing_questions = {
        (ex.inputs or {}).get("messages", [{}])[0].get("content", "")
        for ex in client.list_examples(dataset_id=dataset.id)
    }

for q, a in examples:
    if q in existing_questions:
        continue
    client.create_example(
        inputs={"messages": [{"role": "user", "content": q}]},
        outputs={"expected_answer": a},
        dataset_id=dataset.id,
    )
    existing_questions.add(q)

def predict(inputs: dict) -> dict:
    """Wywołuje workflow: wejście z datasetu (messages), zwraca ostatnią odpowiedź jako final_report."""
    messages = inputs.get("messages", [])
    content = messages[0]["content"] if messages else ""
    lc_messages = [HumanMessage(content=content)]
    result = graph.invoke({"messages": lc_messages})
    answer = result["messages"][-1].content if result.get("messages") else ""
    return {"final_report": answer}


def qa_correctness(run: Run, example: Example) -> dict:
    """Ewaluator LLM-as-judge: model ocenia zgodność odpowiedzi z oczekiwaną (0.0–1.0)."""
    expected = (example.outputs or {}).get("expected_answer", "")
    actual = (run.outputs or {}).get("final_report", "")
    question = (example.inputs or {}).get("messages", [{}])[0].get("content", "")
    if not question or (not expected and not actual):
        return {"key": "qa_correctness", "score": 0.0}
    prompt = EVAL_PROMPT.format(question=question, expected=expected, actual=actual)
    grader = eval_model.with_structured_output(EvalScore)
    result = grader.invoke([{"role": "user", "content": prompt}])
    score = max(0.0, min(1.0, float(result.score)))
    return {"key": "qa_correctness", "score": score}


if __name__ == "__main__":
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[qa_correctness],
        experiment_prefix="initial-test-gpt4o",
    )
    print(results)

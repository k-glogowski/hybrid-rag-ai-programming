"""
Proste GUI Streamlit do uruchamiania ewaluacji RAG i wyświetlania wyników.
"""
import streamlit as st
from langsmith.evaluation import evaluate

# Import logiki ewaluacji z client (dataset, predict, qa_correctness są tworzone przy imporcie)
from client import dataset_name, predict, qa_correctness

st.set_page_config(page_title="Ewaluacja RAG", page_icon="📊", layout="centered")
st.title("📊 Ewaluacja pipeline'u RAG")
st.caption("Uruchom ewaluację na zbiorze Docker RAG i zobacz wyniki (LLM-as-judge).")

experiment_prefix = st.text_input(
    "Prefiks eksperymentu (opcjonalnie)",
    value="eval-run",
    help="Nazwa uruchomienia w LangSmith.",
)

if st.button("▶ Uruchom ewaluację", type="primary"):
    with st.spinner("Trwa ewaluacja (wywołania modelu dla każdego przykładu)..."):
        try:
            results = evaluate(
                predict,
                data=dataset_name,
                evaluators=[qa_correctness],
                experiment_prefix=experiment_prefix or "eval-run",
            )
        except Exception as e:
            st.error(f"Błąd podczas ewaluacji: {e}")
            st.stop()

    st.success("Ewaluacja zakończona.")

    # Nagłówek wyników
    st.subheader("Wyniki")
    if hasattr(results, "experiment_name"):
        st.write(f"**Eksperyment:** `{results.experiment_name}`")

    # Tabela wyników per przykład
    rows = []
    for row in getattr(results, "results", []) or []:
        example = row.get("example") or row.get("example_id")
        evals = row.get("evaluation_results") or []
        if example is None:
            continue
        # example może być obiektem z .inputs, .outputs lub dict
        if hasattr(example, "inputs"):
            inputs = example.inputs or {}
            outputs = example.outputs or {}
        else:
            inputs = (example or {}).get("inputs", {})
            outputs = (example or {}).get("outputs", {})

        messages = inputs.get("messages", [{}])
        question = messages[0].get("content", "") if messages else ""
        expected = outputs.get("expected_answer", "")

        score = None
        comment = ""
        for e in evals if isinstance(evals, list) else [evals]:
            if isinstance(e, dict):
                if e.get("key") == "qa_correctness":
                    score = e.get("score")
                    comment = e.get("comment", "")
                    break
            elif hasattr(e, "key") and e.key == "qa_correctness":
                score = getattr(e, "score", None)
                comment = getattr(e, "comment", "") or ""
                break

        rows.append(
            {
                "Pytanie": question[:80] + ("..." if len(question) > 80 else ""),
                "Oczekiwana odpowiedź": (expected[:60] + "...") if len(expected) > 60 else expected,
                "Score (qa_correctness)": f"{score:.2f}" if score is not None else "—",
                "Komentarz": comment[:100] + ("..." if len(comment) > 100 else "") if comment else "—",
            }
        )

    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        # Podsumowanie
        scores = []
        for r in rows:
            s = r.get("Score (qa_correctness)")
            if s and s != "—":
                try:
                    scores.append(float(s))
                except ValueError:
                    pass
        if scores:
            avg = sum(scores) / len(scores)
            st.metric("Średni score (qa_correctness)", f"{avg:.2f}")
    else:
        st.info("Brak wierszy wyników do wyświetlenia. Sprawdź strukturę zwracaną przez `evaluate()`.")


    with st.expander("Wyniki (szczegóły per przykład)"):
        exp_name = getattr(results, "experiment_name", None) or "—"
        for i, result in enumerate(results):
            run = result.get("run") if isinstance(result, dict) else getattr(result, "run", None)
            example = result.get("example") if isinstance(result, dict) else getattr(result, "example", None)
            eval_res = result.get("evaluation_results") if isinstance(result, dict) else getattr(result, "evaluation_results", None)

            # Experiment name (z obiektu run lub z results)
            run_name = getattr(run, "name", None) if run else None
            experiment_name = run_name or exp_name

            # Input (pytanie z example.inputs)
            if example:
                inputs = getattr(example, "inputs", None) or (example if isinstance(example, dict) else {}).get("inputs", {})
                messages = (inputs or {}).get("messages", [{}])
                input_text = messages[0].get("content", "") if messages else ""
            else:
                input_text = "—"

            # Output (odpowiedź modelu z run.outputs)
            if run:
                outputs = getattr(run, "outputs", None) or (run if isinstance(run, dict) else {}).get("outputs", {})
                output_text = (outputs or {}).get("final_report", "") or "—"
            else:
                output_text = "—"

            # Score z evaluation_results.results (qa_correctness)
            score_val = None
            if eval_res:
                res_list = getattr(eval_res, "results", None) or (eval_res if isinstance(eval_res, dict) else {}).get("results", [])
                for e in res_list or []:
                    key = getattr(e, "key", None) or (e.get("key") if isinstance(e, dict) else None)
                    if key == "qa_correctness":
                        score_val = getattr(e, "score", None) or (e.get("score") if isinstance(e, dict) else None)
                        break

            st.markdown(f"**Wynik {i+1}**")
            st.caption(f"Eksperyment: {experiment_name}")
            st.text_area("Input (pytanie)", value=input_text, key=f"input_{i}", height=80, disabled=True)
            st.text_area("Output (odpowiedź modelu)", value=output_text, key=f"output_{i}", height=220, disabled=True)
            st.metric("Score (qa_correctness)", f"{score_val:.2f}" if score_val is not None else "—")
            st.divider()

import pytest
from slr_agent.subgraphs.pico import create_pico_subgraph, PICOSubgraphState
from slr_agent.llm import MockLLM

@pytest.fixture
def mock_llm():
    llm = MockLLM()
    llm.register("detect the language", {"language_code": "en"})
    llm.register("expand this research question into PICO", {
        "population": "adults with hypertension",
        "intervention": "ACE inhibitors",
        "comparator": "placebo",
        "outcome": "blood pressure reduction",
    })
    llm.register("Generate 3-4 PubMed search query strings", {
        "query_strings": [
            "(ACE inhibitors OR angiotensin converting enzyme inhibitors) AND (blood pressure OR hypertension)",
            "(antihypertensive) AND (hypertension OR high blood pressure) AND (adults)",
        ]
    })
    return llm

def test_pico_subgraph_produces_result(mock_llm):
    graph = create_pico_subgraph(llm=mock_llm)
    initial: PICOSubgraphState = {
        "raw_question": "Do ACE inhibitors reduce blood pressure in hypertensive adults?",
        "pico": None,
        "validation_errors": [],
    }
    result = graph.invoke(initial)
    assert result["pico"] is not None
    assert result["pico"]["population"] == "adults with hypertension"
    assert len(result["pico"]["query_strings"]) == 2
    assert result["validation_errors"] == []

def test_pico_subgraph_detects_non_english():
    from slr_agent.llm import MockLLM
    llm = MockLLM()
    llm.register("detect the language", {"language_code": "fr"})
    llm.register("translate the following research question to English", {
        "translated": "Do ACE inhibitors reduce blood pressure in hypertensive adults?"
    })
    llm.register("expand this research question into PICO", {
        "population": "adults with hypertension",
        "intervention": "ACE inhibitors",
        "comparator": "placebo",
        "outcome": "blood pressure reduction",
    })
    llm.register("Generate 3-4 PubMed search query strings", {
        "query_strings": ["(ACE inhibitors) AND (hypertension OR blood pressure)"]
    })
    graph = create_pico_subgraph(llm=llm)
    initial = {
        "raw_question": "Les inhibiteurs de l'ECA réduisent-ils la pression artérielle?",
        "pico": None,
        "validation_errors": [],
    }
    result = graph.invoke(initial)
    assert result["pico"]["source_language"] == "fr"
    assert result["pico"]["search_language"] == "en"

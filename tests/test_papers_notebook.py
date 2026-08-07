"""Guard the scrub of the committed analysis notebook.

The notebook's outputs contain participant-level study data, so the
committed copy must stay output-free and parameterized — re-committing an
executed copy is a data leak, and this test makes that a red build.
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "papers" / "lak2026" / "analysis.ipynb"


def test_analysis_notebook_is_scrubbed():
    nb = json.loads(NOTEBOOK.read_text())

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [], f"cell {i} has committed outputs"
        assert cell["execution_count"] is None, f"cell {i} has an execution count"

    text = json.dumps(nb)
    # Inputs are parameterized, questions load through the validated reader
    assert "STUDY_DIR" in text and "FIGURES_DIR" in text
    assert "load_questions" in text
    assert "eval(line)" not in text

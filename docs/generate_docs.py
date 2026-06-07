
"""Generate the deliverable PDFs into docs/.

Run from the project root:

    venv\\Scripts\\python docs\\generate_docs.py
"""

import os
import sys
import types

# allow running as a plain script (stub streamlit so report.py imports cleanly)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "streamlit" not in sys.modules:
    st = types.ModuleType("streamlit")
    st.session_state = {}
    st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules["streamlit"] = st

import pandas as pd
from modules.docgen import (build_analysis_doc, build_analysis_docx,
                            build_system_doc, build_system_docx)
from modules.datasets import DEFAULT_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "student_dataset.csv")


def main():
    df = pd.read_csv(DATA)
    name = DEFAULT_NAME

    outputs = {
        "Analysis_Documentation.pdf": build_analysis_doc(df, name),
        "Analysis_Documentation.docx": build_analysis_docx(df, name),
        "System_Documentation.pdf": build_system_doc(df, name),
        "System_Documentation.docx": build_system_docx(df, name),
    }
    for fname, data in outputs.items():
        path = os.path.join(HERE, fname)
        open(path, "wb").write(data)
        print(f"Wrote {path} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()

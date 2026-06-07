
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
from modules.docgen import build_analysis_doc, build_system_doc
from modules.datasets import DEFAULT_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "student_dataset.csv")


def main():
    df = pd.read_csv(DATA)
    name = DEFAULT_NAME

    analysis = build_analysis_doc(df, name)
    apath = os.path.join(HERE, "Analysis_Documentation.pdf")
    open(apath, "wb").write(analysis)
    print(f"Wrote {apath} ({len(analysis):,} bytes)")

    system = build_system_doc(df, name)
    spath = os.path.join(HERE, "System_Documentation.pdf")
    open(spath, "wb").write(system)
    print(f"Wrote {spath} ({len(system):,} bytes)")


if __name__ == "__main__":
    main()

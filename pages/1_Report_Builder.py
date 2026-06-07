
import streamlit as st

st.set_page_config(page_title="Report Builder · Smart Analysis Reporter",
                   page_icon="📝", layout="wide")

from modules import ui
from modules.datasets import ensure_dataset_loaded, DEFAULT_NAME
from modules.autoreport import build_default_report
from modules import report as R

ui.setup()
ensure_dataset_loaded()

ui.header("Report Builder", "Edit the cover, refine every inference, then download your report.", icon="📝")

rep = R.get_report()
cover = rep["cover"]

# ── toolbar ──
t1, t2, t3 = st.columns([1.2, 1, 3])
with t1:
    if st.button("✨ Auto-generate / rebuild", type="primary", use_container_width=True):
        df = st.session_state["df"]
        build_default_report(df, st.session_state.get("dataset_name", DEFAULT_NAME))
        st.rerun()
with t2:
    if st.button("🗑️ Clear report", use_container_width=True):
        R.reset_report()
        st.rerun()
with t3:
    st.caption(f"{len(rep['items'])} section(s) in this report.")

st.divider()

# ── editable cover ──
st.subheader("Cover")

cc1, cc2 = st.columns(2)
with cc1:
    cover["title"] = st.text_input("Report title", cover["title"])
    cover["author"] = st.text_input("Author", cover["author"])
    cover["date"] = st.text_input("Date", cover["date"])
with cc2:
    cover["subtitle"] = st.text_input("Subtitle / dataset", cover["subtitle"])
    cover["author_id"] = st.text_input("Author ID", cover["author_id"])

cover["summary"] = st.text_area(
    "Executive summary (editable — replace with your own words anytime)",
    cover["summary"], height=120,
)

# cover preview
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,{ui.ORANGE},#FF9E40);color:#fff;
                padding:30px 34px;border-radius:16px;margin:6px 0 4px">
      <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;opacity:.9">
        {ui.APP_NAME}</div>
      <div style="font-size:30px;font-weight:800;margin:8px 0 2px">{cover['title']}</div>
      <div style="font-size:16px;opacity:.95">{cover['subtitle']}</div>
      <div style="margin-top:16px;font-size:13.5px">Prepared by <b>{cover['author']}</b>
        &nbsp;·&nbsp; ID {cover['author_id']} &nbsp;·&nbsp; {cover['date']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── items ──
st.subheader("Sections")

if not rep["items"]:
    st.info("No sections yet. Click **Auto-generate** above, or add items from the analysis pages "
            "using their **➕ Add to report** buttons.")
else:
    for i, it in enumerate(rep["items"]):
        with st.container(border=True):
            head, b_up, b_dn, b_rm = st.columns([6, 1, 1, 1])
            with head:
                it["title"] = st.text_input(
                    "Section title", it["title"], key=f"title_{it['id']}",
                    label_visibility="collapsed",
                )
            with b_up:
                if st.button("↑", key=f"up_{it['id']}", use_container_width=True, disabled=i == 0):
                    R.move_item(it["id"], -1); st.rerun()
            with b_dn:
                if st.button("↓", key=f"dn_{it['id']}", use_container_width=True,
                             disabled=i == len(rep["items"]) - 1):
                    R.move_item(it["id"], 1); st.rerun()
            with b_rm:
                if st.button("✕", key=f"rm_{it['id']}", use_container_width=True):
                    R.remove_item(it["id"]); st.rerun()

            pv1, pv2 = st.columns([1, 1])
            with pv1:
                if it.get("image"):
                    st.image(it["image"], use_container_width=True)
                if it.get("table") is not None:
                    st.dataframe(it["table"], use_container_width=True)
            with pv2:
                it["inference"] = st.text_area(
                    "Inference (editable)", it["inference"],
                    key=f"inf_{it['id']}", height=180,
                )

st.divider()

# ── export ──
st.subheader("Download report")

if rep["items"]:
    safe = cover["title"].replace(" ", "_")[:40] or "report"
    d1, d2, d3, _ = st.columns([1, 1, 1, 2])
    with d1:
        st.download_button(
            "⬇️ PDF", data=R.build_pdf(rep),
            file_name=f"{safe}.pdf", mime="application/pdf",
            type="primary", use_container_width=True,
        )
    with d2:
        st.download_button(
            "⬇️ Word (.docx)", data=R.build_docx(rep),
            file_name=f"{safe}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "⬇️ HTML", data=R.build_html(rep),
            file_name=f"{safe}.html", mime="text/html",
            use_container_width=True,
        )
else:
    st.caption("Add at least one section to enable downloads.")

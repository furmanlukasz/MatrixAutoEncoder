import streamlit as st
import pathlib

def abstract():

    st.write("This is a placeholder for the abstract and conclusion section.")
    st.write("Please refer to the results and analysis for detailed information.")

    with open(pathlib.Path(__file__).parent.parent.parent / "doc" / "abstract_page.md", "r") as file:
        st.markdown(file.read(), unsafe_allow_html=False)
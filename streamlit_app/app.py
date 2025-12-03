import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Streamlit + FastAPI", layout="wide")
st.title("Streamlit Panel")
st.caption(f"Backend: {API_BASE}")

if st.button("Ping FastAPI"):
    try:
        r = requests.get(f"{API_BASE}/api/hello", timeout=5)
        st.success(r.json())
    except Exception as e:
        st.error(f"Error: {e}")

name = st.text_input("Name", "Makenna")
if st.button("Send"):
    st.write(f"Hi, {name}! (stub)")

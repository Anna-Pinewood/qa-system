import json

import requests
import streamlit as st

API_URL = "http://0.0.0.0:5055"  # Use docker app name or 0.0.0.0 if running locally


def main():
    st.title("Chatbot")
    user_input = st.text_area("Enter your message here", height=200)
    if st.button("Send"):
        response = requests.post(f"{API_URL}/answer", json={"query": user_input})
        answer = json.loads(response.text)["answer"]
        st.text_area("Response", value=answer, height=200)
    else:
        st.warning("Please enter a message")


if __name__ == "__main__":
    main()

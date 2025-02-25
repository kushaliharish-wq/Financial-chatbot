import streamlit as st
import requests
import uuid

if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []
st.title("Financio - AI Financial Assistant")
st.sidebar.title("Model Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Enter your query"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    payload = {
        "user_id": st.session_state["user_id"],
        "question": prompt,
        "chat_history": [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state["messages"]
        ],
    }

    try:
        response = requests.post("http://localhost:8001/query", json=payload)
        response.raise_for_status()
        data = response.json()
        bot_response = data.get("response", "I'm sorry, I couldn't fetch a response.")
    except requests.exceptions.RequestException as e:
        bot_response = f"Error communicating with the backend: {e}"
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})

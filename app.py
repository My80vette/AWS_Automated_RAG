import streamlit as st
import requests

# Page config
st.set_page_config(page_title="Chat Interface", layout="wide")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("Options")
    # Add your options here as you develop

# Main chat interface
st.title("Kubernetes Technical Support RAG Platform")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("Kubernetes Manager"):
        # Call BentoML service
        try:
            api_response = requests.post(
                # Replace with EKS endpoint URL when bento gets deployed
                "http://localhost:3000/answer_question",
                json={"query": prompt}
            )
            response_data = api_response.json()
            response = response_data.get("answer", "No answer received")
        except Exception as e:
            response = f"Error: {str(e)}"
        
        st.markdown(response)
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "Kubernetes Manager", "content": response})

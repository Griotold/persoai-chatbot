import streamlit as st
import uuid
from dotenv import load_dotenv
from llm import get_ai_response

st.set_page_config(page_title="Perso.ai 챗봇", page_icon="🎬")
st.title("🎬 Perso.ai 챗봇")
st.caption("Perso.ai 서비스에 관련된 모든 것을 답해드립니다!")

load_dotenv()

# 세션 ID 생성 (각 사용자마다 고유)
if 'user_session_id' not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자가 입력한 메시지를 받음 (입력란이 비어있지 않으면 실행)
if user_question := st.chat_input(placeholder="Perso.ai에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
    
    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question, st.session_state.user_session_id)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
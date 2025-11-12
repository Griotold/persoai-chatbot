from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'persoai-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "채팅 기록과 최신 사용자 질문이 주어졌을 때, "
        "채팅 기록의 맥락을 참조할 수 있는 질문을 "
        "채팅 기록 없이도 이해할 수 있는 독립적인 질문으로 바꿔주세요. "
        "질문에 답변하지 말고, 필요한 경우에만 질문을 다시 구성하고 "
        "그렇지 않으면 그대로 반환해주세요."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    return history_aware_retriever

def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 Perso.ai 서비스 전문가입니다. 사용자의 Perso.ai에 관한 질문에 답변해주세요. "
        "아래에 제공된 문서를 활용해서 답변해주시고, "
        "답변을 알 수 없다면 '제공된 정보에서는 해당 내용을 찾을 수 없습니다'라고 답변해주세요. "
        "간결하고 명확하게 2-3 문장 정도로 답변해주세요."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain


def get_ai_response(user_message, session_id):
    rag_chain = get_rag_chain()
    ai_response = rag_chain.stream(
        {
            "input": user_message
        },
        config={
            "configurable": {"session_id": session_id}
        },
    )
    return ai_response
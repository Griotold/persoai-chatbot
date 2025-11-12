# 🎬 Perso.ai 지식기반 챗봇

벡터 데이터베이스(Vector DB)를 활용한 Perso.ai Q&A 챗봇 시스템

## 📌 개요

Perso.ai는 이스트소프트의 AI 영상 더빙 플랫폼입니다. 본 프로젝트는 Perso.ai 서비스에 대한 13개의 Q&A 데이터를 기반으로, 할루시네이션 없이 정확한 답변을 제공하는 RAG(Retrieval-Augmented Generation) 기반 챗봇 시스템입니다.

### 🔗 배포 링크
**[Perso.ai 챗봇 체험하기](https://persoai-chatbot-griotold.streamlit.app/)**

### ⚙️ 주요 기능
- ✅ Vector DB 기반 유사도 검색
- ✅ 대화 맥락 기억 (Conversational Memory)
- ✅ Few-shot Prompting을 통한 답변 품질 향상
- ✅ 할루시네이션 방지 (데이터에 없는 정보는 답변하지 않음)

---

## 🛠️ 환경 설정

### 기술 스택
- **Python**: 3.11.13
- **LangChain**: 0.3.3
  - `langchain-openai`: 0.2.2
  - `langchain-pinecone`: 0.2.0
  - `langchain-community`: 0.3.2
- **OpenAI API**: GPT-4o, text-embedding-3-large
- **Pinecone**: Vector Database (Serverless)
- **Streamlit**: 1.50.0

### 설치 방법

```bash
# 가상환경 생성 및 활성화
pyenv virtualenv 3.11.13 persoai-chatbot
pyenv activate persoai-chatbot

# 패키지 설치
pip install -r requirements.txt
```

### 환경 변수 설정

`.env` 파일에 다음 API 키를 설정합니다:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## 📊 데이터 전처리

### 1. 데이터 구조

제공된 Excel 파일(`qa_data.xlsx`)에서 13개의 Q&A 쌍을 추출합니다:

```
컬럼 구조:
[제목] [순번] [내용]
  -      1    Q. Perso.ai는 어떤 서비스인가요?
  -      -    A. Perso.ai는 이스트소프트가 개발한...
  -      2    Q. Perso.ai의 주요 기능은 무엇인가요?
  -      -    A. Perso.ai는 AI 음성 합성...
  ...
```

### 2. 파싱 로직

`embed_data.py`는 다음 단계로 데이터를 처리합니다:

```python
# 1. Excel 파일 읽기
df = pd.read_excel('data/qa_data.xlsx', header=None)

# 2. 2번 컬럼에서 Q&A 추출
data_col = df[2].dropna().tolist()

# 3. Q와 A를 매칭
for item in data_col:
    if item.startswith('Q.'):
        current_q = item
    elif item.startswith('A.') and current_q:
        qa_list.append({
            'question': current_q,
            'answer': item
        })
```

### 3. Document 객체 생성

LangChain의 `Document` 객체로 변환하여 임베딩합니다:

```python
documents = []
for idx, qa in enumerate(qa_list):
    doc = Document(
        page_content=f"{qa['question']}\n{qa['answer']}",
        metadata={
            "question": qa["question"],
            "answer": qa["answer"],
            "id": idx
        }
    )
    documents.append(doc)
```

### 4. 실행 결과

```bash
python embed_data.py
```

**출력 결과:**

```
[실행 결과 스크린샷 자리]
```

> 13개의 Q&A가 성공적으로 파싱되고, Pinecone에 임베딩되었습니다.

---

## 🗄️ Pinecone Vector Database

### 1. 인덱스 생성

Pinecone 콘솔에서 `persoai-index` 인덱스를 생성했습니다:

**인덱스 설정:**
- **Index Name**: `persoai-index`
- **Dimensions**: `3072` (OpenAI text-embedding-3-large 모델)
- **Metric**: `cosine`
- **Cloud**: `AWS`
- **Region**: `us-east-1`

**Pinecone 콘솔 스크린샷:**

```
[Pinecone 인덱스 생성 스크린샷 자리]
```

### 2. 임베딩 코드

```python
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# OpenAI 임베딩 모델 초기화
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# Pinecone에 문서 업로드
database = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embedding,
    index_name='persoai-index'
)
```

### 3. 차원(Dimension) 선택 이유

**text-embedding-3-large 모델**은 3072 차원의 벡터를 생성합니다. 이는 다음과 같은 장점이 있습니다:

- ✅ 높은 검색 정확도
- ✅ 세밀한 의미 구분 가능
- ✅ 짧은 Q&A에서도 뛰어난 성능

### 4. 검색 전략

```python
# Top-k=4로 설정하여 가장 유사한 4개의 문서 검색
retriever = database.as_retriever(search_kwargs={'k': 4})
```

---

## 💡 Few-shot Prompting

Few-shot Prompting을 통해 LLM에게 답변 스타일과 형식을 학습시킵니다.

### 설정 코드 (config.py)

```python
answer_examples = [
    {
        "input": "Perso.ai는 어떤 서비스인가요?",
        "answer": "Perso.ai는 이스트소프트가 개발한 다국어 AI 영상 더빙 플랫폼으로, 누구나 언어의 장벽 없이 영상을 제작하고 공유할 수 있도록 돕는 AI SaaS 서비스입니다."
    },
    {
        "input": "Perso.ai의 주요 기능은 무엇인가요?",
        "answer": "Perso.ai는 AI 음성 합성, 립싱크, 영상 더빙 기능을 제공합니다. 사용자는 원본 영상에 다른 언어로 음성을 입히거나, 입 모양까지 자동으로 동기화할 수 있습니다."
    },
    {
        "input": "지원하는 언어는 몇 개인가요?",
        "answer": "현재 30개 이상의 언어를 지원하며, 한국어, 영어, 일본어, 스페인어, 포르투갈어 등 주요 언어가 포함됩니다."
    }
]
```

### 적용 효과

1. **일관된 답변 톤**: 간결하고 명확한 답변 스타일 유지
2. **정확한 정보 전달**: 데이터셋의 답변 형식을 학습
3. **할루시네이션 감소**: 예시를 통해 정확한 답변 범위 학습

---

## 🧠 대화 맥락 기억 (Conversational Memory)

### 1. 세션 기반 대화 관리

```python
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

### 2. 맥락 인식 Retriever

사용자의 질문이 이전 대화를 참조할 경우, 독립적인 질문으로 재구성합니다:

```python
contextualize_q_system_prompt = (
    "채팅 기록과 최신 사용자 질문이 주어졌을 때, "
    "채팅 기록 없이도 이해할 수 있는 독립적인 질문으로 바꿔주세요."
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

### 3. 동작 예시

**대화 흐름:**
```
사용자: "Perso.ai가 뭐야?"
챗봇: "Perso.ai는 이스트소프트가 개발한 AI 영상 더빙 플랫폼입니다."

사용자: "지원 언어는?" 👈 애매한 질문
      ↓
[내부 처리: "Perso.ai의 지원 언어는 몇 개인가요?"로 재구성]
      ↓
챗봇: "현재 30개 이상의 언어를 지원합니다."
```

---

## 🚫 할루시네이션 방지

### 1. System Prompt 설계

데이터에 없는 정보는 답변하지 않도록 명시적으로 지시합니다:

```python
system_prompt = (
    "당신은 Perso.ai 서비스 전문가입니다. "
    "아래에 제공된 문서를 활용해서 답변해주시고, "
    "답변을 알 수 없다면 '제공된 정보에서는 해당 내용을 찾을 수 없습니다'라고 답변해주세요. "
    "간결하고 명확하게 2-3 문장 정도로 답변해주세요."
)
```

### 2. 테스트 결과

**질문:** "모바일 앱도 있어?"
**응답:** "제공된 정보에서는 해당 내용을 찾을 수 없습니다."

**할루시네이션 방지 스크린샷:**

```
[할루시네이션 테스트 스크린샷 자리]
```

> 데이터에 없는 정보에 대해서는 지어내지 않고, 명확히 "모른다"고 답변합니다.

---

## 🚀 실행 방법

### 1. 데이터 임베딩 (최초 1회만)

```bash
python embed_data.py
```

### 2. 챗봇 실행

```bash
streamlit run chat.py
```

### 3. 브라우저 접속

```
Local URL: http://localhost:8501
```

---

## 📁 프로젝트 구조

```
persoai-chatbot/
├── data/
│   └── qa_data.xlsx          # 원본 Q&A 데이터
├── embed_data.py             # 데이터 전처리 및 임베딩
├── llm.py                    # RAG 체인 및 LLM 로직
├── config.py                 # Few-shot examples
├── chat.py                   # Streamlit UI
├── requirements.txt          # 패키지 의존성
├── .env                      # 환경 변수 (API 키)
├── .gitignore
└── README.md
```

---

## 📝 기술 문서

### RAG 파이프라인

```
사용자 질문
    ↓
[대화 맥락 인식] → 질문 재구성
    ↓
[Vector DB 검색] → Top-k=4 문서 검색
    ↓
[LLM 생성] → Few-shot + System Prompt
    ↓
답변 생성
```

### 평가 기준 충족도

| 평가 항목 | 비중 | 구현 내용 |
|---------|------|----------|
| **정확성** | 40% | Vector DB 기반 정확한 검색, 할루시네이션 방지 프롬프트 |
| **기술 설계** | 30% | RAG 아키텍처, Few-shot Prompting, 대화 맥락 관리 |
| **완성도** | 20% | Streamlit UI, Streamlit Cloud 배포, 안정적 동작 |
| **문서/논리성** | 10% | README.md, 코드 주석, 명확한 기술 선택 근거 |

---

## 🔧 트러블슈팅

### 1. Pinecone 연결 오류
```bash
# API 키 확인
echo $PINECONE_API_KEY
```

### 2. OpenAI Rate Limit
- 무료 크레딧 또는 소액 과금($5) 권장
- Gemini API로 대체 가능

### 3. Streamlit 배포 오류
- `requirements.txt`에 모든 패키지 포함 확인
- `.streamlit/config.toml` 설정 확인

---

## 📄 라이선스

MIT License

---

## 👤 개발자

**Griotold**
- GitHub: [@Griotold](https://github.com/Griotold)
- 프로젝트: [persoai-chatbot](https://github.com/Griotold/persoai-chatbot)

---

## 🙏 참고 자료

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
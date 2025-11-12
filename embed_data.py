import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone

# 환경 변수 로드
load_dotenv()

def load_and_check_data():
    """1. xlsx 파일 읽기 및 확인"""
    print("\n[STEP 1] 📂 데이터 로딩 중...")
    
    file_path = 'data/qa_data.xlsx'
    df = pd.read_excel(file_path, header=None)
    
    print(f"✅ 파일 로드 완료: {file_path}")
    print(f"📊 전체 행 수: {len(df)}")
    
    # 데이터 미리보기
    print("\n=== 데이터 미리보기 (처음 10행) ===")
    print(df.head(10))
    print("\n=== 데이터 미리보기 (마지막 10행) ===")
    print(df.tail(10))
    
    return df

def parse_qa_data(df):
    """2. Q&A 데이터 파싱"""
    print("\n[STEP 2] 🔍 Q&A 파싱 중...")
    
    # 실제 데이터는 컬럼 2에 있음
    data_col = df[2].dropna().tolist()
    
    qa_list = []
    current_q = None
    
    for item in data_col:
        item_str = str(item)
        if item_str.startswith('Q.'):
            current_q = item_str
        elif item_str.startswith('A.') and current_q:
            qa_list.append({
                'question': current_q,
                'answer': item_str
            })
            current_q = None
    
    print(f"✅ 파싱 완료: {len(qa_list)}개의 Q&A 쌍")
    
    # 파싱된 데이터 확인
    print("\n=== 파싱된 Q&A 데이터 확인 ===")
    for i, qa in enumerate(qa_list[:3], 1):  # 처음 3개만 출력
        print(f"\n[{i}]")
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer'][:80]}...")  # 답변은 80자까지만
    
    if len(qa_list) > 3:
        print(f"\n... (나머지 {len(qa_list) - 3}개 생략)")
    
    return qa_list

def create_documents(qa_list):
    """3. Document 객체 생성"""
    print("\n[STEP 3] 📝 Document 객체 생성 중...")
    
    documents = []
    for idx, qa in enumerate(qa_list):
        # 질문과 답변을 함께 page_content에 저장
        doc = Document(
            page_content=f"{qa['question']}\n{qa['answer']}",
            metadata={
                "question": qa["question"],
                "answer": qa["answer"],
                "id": idx
            }
        )
        documents.append(doc)
    
    print(f"✅ Document 생성 완료: {len(documents)}개")
    
    return documents

def embed_to_pinecone(documents):
    """4. Pinecone에 임베딩 & 업로드"""
    print("\n[STEP 4] 🚀 Pinecone에 임베딩 중...")
    
    # 임베딩 모델 초기화
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    
    # Pinecone 설정
    index_name = 'persoai-index'
    
    # Pinecone 인덱스 존재 확인
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"❌ 인덱스 '{index_name}'가 존재하지 않습니다.")
        print("Pinecone 콘솔에서 인덱스를 먼저 생성해주세요.")
        return None
    
    print(f"✅ 인덱스 확인: {index_name}")
    
    # Pinecone에 임베딩 & 저장
    print(f"⏳ 임베딩 중... (약 10-20초 소요)")
    database = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        index_name=index_name
    )
    
    print(f"✅ 임베딩 완료! {len(documents)}개의 문서가 Pinecone에 저장되었습니다.")
    
    return database

def verify_embeddings():
    """5. 임베딩 검증"""
    print("\n[STEP 5] 🔍 임베딩 검증 중...")
    
    # 임베딩 & 인덱스 로드
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'persoai-index'
    
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )
    
    # 테스트 쿼리
    test_queries = [
        "Perso.ai는 무엇인가요?",
        "지원하는 언어는?",
        "요금제는 어떻게 되나요?"
    ]
    
    print("\n=== 검증 테스트 ===")
    for query in test_queries:
        print(f"\n📝 테스트 쿼리: '{query}'")
        results = database.similarity_search(query, k=2)
        
        if results:
            print(f"✅ 검색 결과 {len(results)}개 발견")
            print(f"   가장 유사한 질문: {results[0].metadata['question']}")
        else:
            print("❌ 검색 결과 없음")
    
    print("\n✅ 검증 완료!")

def main():
    """전체 프로세스 실행"""
    print("=" * 80)
    print("🚀 Perso.ai Q&A 임베딩 스크립트")
    print("=" * 80)
    
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY가 .env 파일에 없습니다.")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY가 .env 파일에 없습니다.")
        return
    
    try:
        # 1. 데이터 로드 & 확인
        df = load_and_check_data()
        
        # 2. Q&A 파싱
        qa_list = parse_qa_data(df)
        
        # 3. Document 생성
        documents = create_documents(qa_list)
        
        # 4. Pinecone에 임베딩
        database = embed_to_pinecone(documents)
        
        if database:
            # 5. 검증
            verify_embeddings()
            
            print("\n" + "=" * 80)
            print("🎉 완료! 이제 챗봇을 실행할 수 있습니다.")
            print("💡 실행 명령어: streamlit run chat.py")
            print("=" * 80)
        
    except FileNotFoundError:
        print("\n❌ Error: data/qa_data.xlsx 파일을 찾을 수 없습니다.")
        print("파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"\n❌ Error 발생: {e}")
        raise

if __name__ == "__main__":
    main()
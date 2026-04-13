import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# تحميل متغيرات البيئة
load_dotenv()

CHROMA_DB_DIR = "../data/chroma_db"

def format_docs(docs):
    # دالة لترتيب النصوص المستخرجة عشان الـ LLM يفهمها بسهولة
    return "\n\n".join(doc.page_content for doc in docs)

def get_compliance_agent():
    # 1. ربط الـ Agent بقاعدة البيانات
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # اختيار أفضل 5 قطع نصوص لها علاقة بالسؤال
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12, 
            "fetch_k": 30,
            "lambda_mult": 0.7  # موازنة بين دقة البحث وتنوع المعلومات
        }
    ) 

    # 2. إعداد العقل المدبر (نستخدم GPT-4o-mini)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3. هندسة الأوامر (The Brain Co. Prompt)
    system_prompt = (
        "You are a sovereign regulatory expert engineer specializing in the Abu Dhabi International Building Code.\n"
        "Your task is to review project proposals or engineering inquiries and check their compliance based ONLY on the provided context.\n\n"
        "Strict Instructions:\n"
        "1. Analyze the inquiry and answer clearly based on the provided text.\n"
        "2. Explain your reasoning clearly from an engineering and legal perspective.\n"
        "3. You MUST document your answer by explicitly citing the (Page Number) extracted from the context.\n"
        "4. If you cannot find the answer in the provided context, state clearly: 'There is not enough information in the provided code.' Do NOT hallucinate or invent an answer.\n\n"
        "Extracted Context:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. بناء الـ Pipeline باستخدام LCEL (الطريقة الأحدث والأسرع)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    # تجربة الـ Agent
    agent = get_compliance_agent()
    
    print("🤖 The Compliance Architect is ready!")
    print("-" * 50)
    
    # سؤال هندسي للتجربة
    test_query = "What are the fire resistance requirements for external walls?"
    
    print(f"👤 السؤال: {test_query}")
    print("⏳ جاري البحث والتحليل...\n")
    
    # نمرر السؤال مباشرة هنا
    response = agent.invoke(test_query)
    
    print("✅ الرد التنظيمي:")
    print(response)
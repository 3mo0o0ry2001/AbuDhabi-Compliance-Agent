import os
import fitz  # PyMuPDF
import logging
from typing import List
from dotenv import load_dotenv

# التحديثات الجديدة لهيكلة LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
# إعداد الـ Logging لتتبع سير العملية باحترافية
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# تحميل متغيرات البيئة (مثل OPENAI_API_KEY)
load_dotenv()

# إعدادات المسارات
PDF_PATH = "../data/abu_dhabi_building_code.pdf"
CHROMA_DB_DIR = "../data/chroma_db"

def extract_text_with_metadata(pdf_path: str) -> List[Document]:
    """
    يستخرج النص من ملف PDF صفحة بصفحة، ويرفقه ببيانات وصفية (Metadata)
    مثل رقم الصفحة، وهو أمر حاسم للـ Agent لتوثيق مصادره.
    """
    logger.info(f"Starting text extraction from: {pdf_path}")
    documents = []
    
    try:
        # فتح ملف الـ PDF
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            
            # تنظيف النص المبدئي (اختياري حسب جودة الملف)
            clean_text = text.replace('\n', ' ').strip()
            
            if clean_text:
                # إنشاء كائن Document الخاص بـ LangChain مع الـ Metadata
                doc_obj = Document(
                    page_content=clean_text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page_number": page_num + 1  # +1 لأن الترقيم يبدأ من 0
                    }
                )
                documents.append(doc_obj)
                
        logger.info(f"Successfully extracted {len(documents)} pages.")
        return documents
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    تقسيم الصفحات إلى قطع (Chunks) أصغر للحفاظ على السياق 
    وتحسين دقة البحث (Retrieval) للـ Agent.
    """
    logger.info("Starting document chunking...")
    
    # نستخدم Recursive Splitter لأنه يحاول الحفاظ على الفقرات والجمل كاملة
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # حجم القطعة التقريبي (بالحروف)
        chunk_overlap=150,  # التداخل بين القطع لعدم فقدان السياق بينها
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def build_vector_database(chunks: List[Document], persist_dir: str):
    """
    تحويل النصوص إلى Vectors باستخدام نماذج التضمين (Embeddings)
    وحفظها في قاعدة بيانات Chroma للاستخدام لاحقاً.
    """
    logger.info("Initializing Embeddings and building Vector Database...")
    
    # نستخدم OpenAI Embeddings (يمكنك استبدالها بـ HuggingFace للمجانية)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    try:
        # إنشاء قاعدة البيانات المتجهة وحفظها محلياً
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        logger.info(f"Vector Database successfully built and saved at: {persist_dir}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error building vector database: {e}")
        raise

if __name__ == "__main__":
    # مسار التنفيذ الرئيسي
    try:
        # 1. الاستخراج
        raw_documents = extract_text_with_metadata(PDF_PATH)
        
        # 2. التقسيم
        document_chunks = chunk_documents(raw_documents)
        
        # 3. التخزين
        build_vector_database(document_chunks, CHROMA_DB_DIR)
        
        logger.info("✅ Ingestion Pipeline Completed Successfully!")
        
    except Exception as e:
        logger.error("❌ Pipeline failed.")
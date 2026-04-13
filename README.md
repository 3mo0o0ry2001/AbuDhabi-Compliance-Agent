# 🏗️ Sovereign Compliance Agent

An Applied AI project designed to automate the review of engineering proposals against the **Abu Dhabi International Building Code** using Agentic RAG (Retrieval-Augmented Generation).

## 🚀 Features
- **Accurate Retrieval:** Uses Advanced RAG (with MMR) to extract precise legal clauses.
- **Sovereign Guardrails:** Strictly answers based on provided regulations without hallucination.
- **Citation-Driven:** Returns the exact page number of the building code for legal trust.
- **Interactive UI:** A clean, production-ready Streamlit dashboard.

## 🛠️ Tech Stack
- **Frameworks:** LangChain, LangGraph (LCEL)
- **Models:** OpenAI `gpt-4o-mini` & `text-embedding-3-small`
- **Vector Database:** ChromaDB
- **UI:** Streamlit
- **Document Processing:** PyMuPDF

## ⚙️ How to Run Locally
1. Clone the repo:
   `git clone https://github.com/YOUR_USERNAME/Sovereign-Compliance-Agent.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Add your OpenAI API key in a `.env` file:
   `OPENAI_API_KEY=sk-...`
4. Place your PDF in the `data/` folder and run the ingestion script.
5. Start the UI:
   `python -m streamlit run src/app.py`

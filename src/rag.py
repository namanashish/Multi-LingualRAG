import os
import sys
sys.path.append("C:/Users/naman/Downloads/Projects/MulitLingualRAG/src")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from parser import DocumentParser

class RAGPipeline:
    def __init__(self, docs_folder: str, db_folder: str = "./chroma_db"):
        self.docs_folder   = docs_folder
        self.db_folder     = db_folder
        self.parser        = DocumentParser(poppler_path=r"C:\poppler\Library\bin")
        self.splitter      = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200,
                                separators=["\n\n", "\n", ".", " "]
                             )
        print("Loading embedding model...")
        self.embeddings    = HuggingFaceEmbeddings(
                                model_name="all-MiniLM-L6-v2",
                                model_kwargs={"device": "cpu"}
                             )
        self.vectorstore   = None
        print("RAG pipeline ready!")

    def ingest(self):
        
        all_chunks = []
        all_metadata = []

        pdf_files = [f for f in os.listdir(self.docs_folder) if f.endswith(".pdf")]
        print(f"Found {len(pdf_files)} PDFs: {pdf_files}")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.docs_folder, pdf_file)
            print(pdf_file)
            pages = self.parser.parse(pdf_path)

            for page in pages:
                if not page["text"].strip():
                    continue

                chunks = self.splitter.split_text(page["text"])

                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source":   pdf_file,
                        "page":   page["page"],
                        "type":   page["type"],
                        "chunk_id": i
                    })

        print(f"Total chunks created: {len(all_chunks)}")

        self.vectorstore = Chroma.from_texts(
            texts=all_chunks,
            embedding=self.embeddings,
            metadatas=all_metadata,
            persist_directory=self.db_folder
        )
        print(f"Stored in ChromaDB at: {self.db_folder}")

    def load(self):
        """Load existing ChromaDB instead of re-ingesting."""
        self.vectorstore = Chroma(
            persist_directory=self.db_folder,
            embedding_function=self.embeddings
        )
        print(f"Loaded ChromaDB from: {self.db_folder}")

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve top_k relevant chunks for a query."""
        if not self.vectorstore:
            raise ValueError("No vectorstore loaded. Run ingest() or load() first.")

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        retrieved = []
        for doc, score in results:
            retrieved.append({
                "text":     doc.page_content,
                "source":   doc.metadata.get("source"),
                "page":     doc.metadata.get("page"),
                "score":    round(score, 4)
            })

        return retrieved

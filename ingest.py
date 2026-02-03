import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pinecone
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def ingest_documents(docs_folder: str):
    """Main ingestion pipeline: PDF -> chunks -> embeddings -> Pinecone."""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize embedding model
    embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME")
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
    
    # Initialize Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    
    # Process each PDF in the folder
    for filename in os.listdir(docs_folder):
        if not filename.lower().endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(docs_folder, filename)
        print(f"\nProcessing: {filename}")
        
        # Step 1: Extract text
        text = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(text)} characters")
        
        # Step 2: Chunk the text
        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings and prepare for upsert
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            vector_id = f"{filename}_{uuid.uuid4().hex[:8]}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": filename,
                    "chunk_index": i
                }
            })
        
        # Step 4: Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"  Uploaded batch {i // batch_size + 1}")
        
        print(f"  Done: {filename}")
    
    print("\nIngestion complete!")

if __name__ == "__main__":
    # Default to a 'docs' folder in the project directory
    docs_path = os.path.join(os.path.dirname(__file__), "docs")
    
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print(f"Created '{docs_path}' folder. Add your PDF files there and run again.")
    else:
        ingest_documents(docs_path)

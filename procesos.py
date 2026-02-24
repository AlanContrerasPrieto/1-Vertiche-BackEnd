import os
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader


def extract_text(directory_path):
        pdf_texts = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                try:
                    with open(pdf_path, 'rb') as file:
                        reader = PdfReader(file)
                        text = ''
                        for page in reader.pages:
                            text += page.extract_text()
                    pdf_texts[filename] = text
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        return pdf_texts


def load_extract_split_embeddings(
    directory_path,
    persist_directory="./chroma_db",
    chunk_size=500,
    chunk_overlap=50,
    model_name="llama3.2",
    create_new=True
):
    # 1. Cargar PDFs
    loader = DirectoryLoader(
        path=directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documentos = loader.load()

    # 2. Extraer texto (opcional si ya cargaste con PyPDFLoader, puedes omitir si prefieres)
    # Si quieres usar la función de extracción de texto manual, descomenta lo siguiente
    
    textos = extract_text(directory_path)
    docs = [{'page_content': t} for t in textos.values()]
    documentos = [Document(page_content=d['page_content'], metadata=d.get('metadata', {})) for d in docs]
    
    # 3. Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documentos)

    # 4. Crear embeddings
    embeddings = OllamaEmbeddings(model=model_name)

    # 5. Crear o cargar la base de datos
    if create_new:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    return vectorstore
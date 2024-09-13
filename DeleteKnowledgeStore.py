from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from chromadb.api.client import SharedSystemClient

#chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())

def deleteVectorsusingKnowledgeBaseID(knowledgeBaseID):
    chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())
    documents = chroma_db.get()
    count=0
    for document_id, metadata in zip(documents['ids'], documents['metadatas']):
        if str(metadata['knowledge_Store_id']) == str(knowledgeBaseID):
            chroma_db.delete([document_id])
            count+=1
    chroma_db._client._system.stop()
    SharedSystemClient._identifer_to_system.pop(chroma_db._client._identifier, None)
    chroma_db = None
    return count>0

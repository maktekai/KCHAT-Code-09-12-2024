import ast
from google import generativeai as genai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Dict, List, Optional, Sequence
from pydantic import BaseModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain_openai import ChatOpenAI
from collections import defaultdict
from chromadb.api.client import SharedSystemClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import os
safety_settings_NONE = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
embedding_function = OpenAIEmbeddings()
RESPONSE_TEMPLATE = """
<context>
    {context}
<context/>

You are a Helpful AI Assistant and Your Task is to Generate a comprehensive and informative answer of 80 words or less for the given question based on the provided context Only  Donot Answer any thing from your own Knowledge Just refer to the COntext Block.
first Search the relevant answer in the Context and then answer the user question.

Must Follow the Below Formatting Instructions:
1. You should use bullet points in your answer for readability. 
2. Put citations where they apply rather than putting them all at the end. Must ensure that you add the citations for each of the Relevenat Answer block.
3. Use proper alignment and indentation and make the format of the answer in the most suitable way.
4. you should provide a answer either in different paragraphs, bullet points or Tables where applicable.

Must Follow the Below Response Instructions:
1. You must only use information from the provided Context Information. 
2. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. 
3. Combine search results together into a coherent answer.
4. Must Cite your answer  using seperate {{Doc ID}} and {{Time Stamp}} notations for reference to exact Context Chunk. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end.
5. If different results refer to different entities within the same name, write separate answers for each entity.
6. If there is nothing in the context relevant to the question at hand, just say "I donot have any information about it because it isn't provided in my context i do apologize for in convenience." Don't try to make up an answer.
7. Anything between the following 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
8. Respond to the Greeting Messages Properly.

Restricted Instruction: Donot Forget to Cite the Answers as Described.
Answer in Markdown:

"""
schema = {
    "properties": {
        "Document_ids_list": {
            "type": "string",
            "description": "Comma Seperated Document IDs as Integers mentioned in the Given Text with out Repetation",
        },
    },
    "required": ["Document_ids_list"],
}

llm_for_tagging = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
chain = create_tagging_chain(schema, llm_for_tagging)

def format_docs(docs: Sequence[Document]) -> List[Document]:
    documents_by_headline = defaultdict(list)
    for doc in docs:
        headline = doc.metadata.get('KuratedContent_headline')
        documents_by_headline[headline].append(doc)
    
    formatted_docs = []
    for i, (headline, document_list) in enumerate(documents_by_headline.items()):
        concatenated_contents = "\n".join([f"{headline}\n{item.page_content}" for item in document_list])
        metadta=document_list[0].metadata
        metadta['doc_id']=i
        formatted_doc = Document(page_content=f"<Document doc_id={i}>{concatenated_contents}<document>", metadata=metadta)
        formatted_docs.append(formatted_doc)
    
    return formatted_docs

def format_references(docs: Sequence[Document]) -> str:
    refrence_docuemnts_sources = []
    for i, doc in enumerate(docs):
        refrence_docuemnts_sources.append({
                'Context-Information': doc.page_content,
                'Source Link': doc.metadata.get('KuratedContent_sourceUrl', ''),
                'Word Press Popup Link': doc.metadata.get('KuratedContent_WordpressPopupUrl', ''),
                'HeadLine': f"<{i}>     "+ doc.metadata.get('KuratedContent_headline','')
            })
    return refrence_docuemnts_sources

def serialize_history(request):
    converted_chat_history = []
    for message in request:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history

def Get_Conversation_chain(knowledgeBases,temperature,model,question,chat_history):
    if model == 'gemini-pro':
        llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True,temperature=float(temperature))
        llm.client = genai.GenerativeModel(model_name='gemini-pro', safety_settings=safety_settings_NONE)
    else:
        llm = ChatOpenAI(
        model=model,
        streaming=True,
        temperature=float(temperature),)
    

    chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())
   # retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
    knowledgeBase=ast.literal_eval(knowledgeBases)
    retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6,"filter":{'knowledge_Store_id': {'$in': (knowledgeBase)}}})
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
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
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever | format_docs, question_answer_chain)
    response=rag_chain.invoke({"input": question, "chat_history": serialize_history(chat_history)})
    answer = response['answer']
#    print("_-----------------------------------",len(response['context']))
 #   print(response['context'])
    refrence_docuemnts_sources=format_references(response['context'])
    result= chain.run(answer)
    #print(result)
    chroma_db._client._system.stop()
    SharedSystemClient._identifer_to_system.pop(chroma_db._client._identifier, None)
    chroma_db = None

    refrence_docuemnts_sources=format_references(response['context'])
    try:
        docuemnts_ids = [int(x) for x in result['Document_ids_list'].split(',')] if result['Document_ids_list'] else []
        filtered_documents = [refrence_docuemnts_sources[idx] for idx in docuemnts_ids if idx < len(refrence_docuemnts_sources)]
        return answer,filtered_documents
    except:
        return answer, refrence_docuemnts_sources

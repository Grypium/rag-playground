#!/mnt/g/workspace/ai/venv/bin/python3.11

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

class PDFChat:
   #vector_store = None
   #retriever = None
   #chain = None

   def __init__(self):
      self.vector_store = None
      self.retriever = None
      self.chain = None

      self.model = ChatOllama(model="llama3.1")
      self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 100)

      self.prompt = PromptTemplate.from_template(
            """
               <s> [INST] You are an assistant specializing in answering questions. Use the provided context to formulate your response. If you are unsure of the answer, state that you don't know. Keep your responses concise, using no more than three sentences. [/INST] </s> 
               [INST] Question: {question} 
               Context: {context} 
               Answer: [/INST]
            """
      )


   def ingest(self, pdf_file_path: str):
      docs = PyPDFLoader(file_path = pdf_file_path).load()

      chunks = self.text_splitter.split_documents(docs)
      chunks = filter_complex_metadata(chunks)

      # WA to resolve initialization problems if passed directly
      embeddings = FastEmbedEmbeddings()
      self.vector_store = Chroma.from_documents(chunks, embeddings, persist_directory = 'chroma_db')
      self.retriever = self.vector_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {
               "k": 3,
               "score_threshold": 0.3,
            }
      )

      self.chain = ({'context': self.retriever, "question": RunnablePassthrough()} | self.prompt | self.model | StrOutputParser())


   def ask(self, query: str):
      if not self.chain:
         return "Please add a document first"

      return self.chain.invoke(query)

   def clear_data(self):
      self.vector_store = None
      self.retriever = None
      self.chain = None


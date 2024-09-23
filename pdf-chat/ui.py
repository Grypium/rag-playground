#!/mnt/g/workspace/ai/venv/bin/python3.11

import os
import tempfile
import streamlit as sl
from streamlit_chat import message
from rag import PDFChat

sl.set_page_config(page_title = "PDF Chat")

def display_messages():
   sl.subheader("Chat")
   
   for i, (msg, is_user) in enumerate(sl.session_state['messages']):
      message(msg, is_user = is_user, key = str(i))
   
   sl.session_state['thinking_spinner'] = sl.empty()

def process_input():
   user_text = sl.session_state.get('user_input', False)
   if user_text:
      user_text = user_text.strip()
      if len(user_text) > 0:
         with sl.session_state['thinking_spinner'], sl.spinner("Thinking.."):
            agent_text = sl.session_state['assistant'].ask(user_text)

         sl.session_state['messages'].append((user_text, True))
         sl.session_state['messages'].append((agent_text, False))


def read_and_save_file():
   sl.session_state['assistant'].clear_data()
   sl.session_state['messages'] = []
   sl.session_state['user_input'] = ""

   for file in sl.session_state['file_uploader']:
      with tempfile.NamedTemporaryFile(delete = False) as temp:
         temp.write(file.getbuffer())
         file_path = temp.name

      with sl.session_state['ingestion_spinner'], sl.spinner(f"Ingesting {file.name}"):
         sl.session_state['assistant'].ingest(file_path)
      os.remove(file_path)

def page():
   if len(sl.session_state) == 0:
      sl.session_state['messages'] = []
      sl.session_state['assistant'] = PDFChat()

   sl.header("PDF Chat")
   sl.subheader("Upload a PDF")
   
   sl.file_uploader(
         "Upload PDF",
         type = ['pdf'],
         key = 'file_uploader',
         on_change = read_and_save_file,
         label_visibility = "collapsed",
         accept_multiple_files = True
   )

   sl.session_state['ingestion_spinner'] = sl.empty()

   display_messages()
   sl.text_input("Message", key = "user_input", on_change = process_input)

if __name__ == "__main__":
   page()

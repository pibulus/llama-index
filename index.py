import os
import streamlit as st
import langchain

from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    download_loader,
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext
)
from langchain import OpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
Airtable_TOKEN = os.getenv('AIRTABLE_TOKEN')

AirtableReader = download_loader('AirtableReader')
reader = AirtableReader(Airtable_TOKEN)
documents = reader.load_data(table_id="tblP6LQxyOo7JBiJC",base_id="appUkRhauFCWrTrBd")

openai.api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

st.title("Ask Llama")
query = st.text_input("What would you like to ask? (source: Airtable)", "")

if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            airtable = Airtable('appUkRhauFCWrTrBd', 'tblP6LQxyOo7JBiJC', api_key=Airtable_TOKEN)
            documents = airtable.get_all()

            # define prompt helper
            max_input_size = 4096
            num_output = 256
            max_chunk_overlap = 20

            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-003", max_tokens=num_output))

            index = GPTVectorStoreIndex.from_documents(documents)

            response = index.query(query)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

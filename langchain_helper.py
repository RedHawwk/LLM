from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import create_engine
from langchain_experimental.sql import SQLDatabaseChain

from urllib.parse import quote


from few_shots import few_shots

import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)


def get_few_shot_db_chain():
    db_user = "root"
    db_password = quote("Kohli@98")  # URL-encode the password
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)
    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
    model_name1 = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(model_name = model_name1)

    to_vectorize = [''.join(qns.values()) for qns in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore = vectorstore,
        k  = 2,
    )

    example_prompt = PromptTemplate(
        input_variables = ['Question','SQLQuery','SQLResult','Answer'],
        template = "\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = _mysql_prompt,
        suffix = PROMPT_SUFFIX,
        input_variables = ["input", "table_info", "top_k"],
    )

    chain = SQLDatabaseChain.from_llm(llm, db, verbose = True, prompt = few_shot_prompt)
    return chain

if __name__ == "__main__":
    chain = get_few_shot_db_chain()
    print(chain.run("How many t-shirts are there of red color?"))
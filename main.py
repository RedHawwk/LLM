from langchain_helper import get_few_shot_db_chain

import streamlit as st
st.title("Koushik T-Shirt: Database Q&A 👕")
st.subheader("Some Query exmaples")

st.caption('eg.1. How many Red color t-shirts are there in the inventory?')
st.caption('eg.2. Number of Medium size t-shirts of adidas in black color')




question = st.text_input("Questions: ")


if question:
    chain = get_few_shot_db_chain()
    answer = chain.run(question)
    st.header("Answer:--")
    st.write(answer)
    # st.header(answer['SQLQuery'])
else:
    st.error("Please enter your query")
import streamlit as st
import pandas as pd 
import retrieve 

# To run: streamlit run app.py

db = retrieve.load_db(None)
retriever = db.as_retriever()

st.title('Skatteregler, avmystifierade')
st.subheader('Din guide i skattedjungeln')

#st.write("Vad är det du försöker ta reda på?")

# Take textual input from the user
user_input = st.text_input("Vad är det du vill ta reda på?")

# Check if there's input before proceeding
if user_input:
    response = retrieve.chat_with_documents(user_input, retriever)
    st.write(f"Svar: {response}")

#st.write(pd.DataFrame({
#    'first column': [1, 2, 3, 4],
#    'second column': [10, 20, 30, 40]
#}))
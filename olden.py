import streamlit as st
import pandas as pd
from transformers import pipeline
import torch
import torch_scatter

def vendor(question): 
    tqa = pipeline(task="table-question-answering", 
                   model="google/tapas-large-finetuned-sqa")
    table = pd.read_csv("data/CP - Copy.csv",encoding='cp1252')
    table = table.astype(str)
    table=table.iloc[0:3, 1:57]
    query = question
    return tqa(table=table, query=query)["answer"]    

if __name__ == '__main__':
    st.title('CharterParty Q&A System')
    st.subheader("This system helps to ask Questions related to the Charter party Contracts")
    #question = st.text_input('Question')     
    title = st.text_input('Please Input your Question', 'Trading exclusion for Cape mathilde')
    df = vendor(title)
    st.write(df)
    st.write("Vessel Name : Annabel : [Annabel CP](https://s3.ap-south-1.amazonaws.com/sm2.0-ap-south-1-274743989443/GPT-3-BM25+MANUAL+EXTRACTION/CP's/Annabel+L+CP.pdf)")
    st.write("Vessel Name : Cape Mathilde :  [Cape Mathilde CP](https://s3.ap-south-1.amazonaws.com/sm2.0-ap-south-1-274743989443/GPT-3-BM25+MANUAL+EXTRACTION/CP's/46346_40_CapeMathildeCPMainBody.pdf)") 
    st.write("Vessel Name : CS Serenity : [CS Serenity CP](https://s3.ap-south-1.amazonaws.com/sm2.0-ap-south-1-274743989443/GPT-3-BM25+MANUAL+EXTRACTION/CP's/Export-38732-NYPE46-CharterParty-2112130619.pdf)")		

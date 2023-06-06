import streamlit as st
from streamlit_chat import message
st.set_page_config(
    page_title="RETALLM",
    page_icon=":robot:"
)
import json
from typing import List, Optional, Tuple, Union

from config import *
from load_model import get_model

def predict(input, history=[], history_rewrite_input=[], history_url=[]):
    request_rewriter, searcher, passage_extractor, answer_generator, fact_checker = get_model()

    #STEP 1 Query Rewriting
    revised_request = request_rewriter.request_rewrite(history_rewrite_input, input)
    history_rewrite_input.append(revised_request) #record all the revised request
    
    #STEP 2 Doc Retrieval
    raw_reference_list = searcher.search(revised_request)
    
    #collect the retrieved urls
    urls = ""
    for raw_reference in raw_reference_list :
        urls = urls + "- {}: {}\n".format(raw_reference['title'], raw_reference['url']) 
    #recorde all historical retrieved urls
    history_url.append(urls) 
    
    #STEP 3 Passage Extractor
    reference = ""
    for raw_reference in raw_reference_list:
        reference = reference + passage_extractor.extract(raw_reference, revised_request, if_extract) + "\n"
    #truncate the references
    reference = reference[:cutoff] 

    with container:
        #display historical conversation
        if len(history) > 0:
            for i, (query, reference, response) in enumerate(history):
                response = response + "\n" +"Reference URL:\n" + history_url[i]
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI is generating response:")

        with st.empty():
            #STEP 4 Answer Generation
            output = answer_generator.answer_generate(reference, revised_request)
            
            #STEP 5 Fact Verification
            if (fact_checker.fact_check(reference, output)) :
                st.write(output)
            else :
                output = "I cannot answer this request."
                st.write(output)

        history.append((input, reference, output))
        st.write("##### Reference URL:\n" + urls)
    return history, history_rewrite_input, history_url


container = st.container()
prompt_text = st.text_area(label="User request input",
            height = 100,
            placeholder="Please input request here.")


if 'state' not in st.session_state:
    st.session_state['state'] = ([],[],[])

if st.button("send", key="predict"):
    with st.spinner("AI is thinking, please wait........"):
        st.session_state["state"] = predict(prompt_text, st.session_state["state"][0], st.session_state["state"][1], st.session_state["state"][2])

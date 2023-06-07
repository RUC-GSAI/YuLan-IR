from config import *
from model_response import generate_response
import streamlit as st
# This is the request rewriter module. 
# We feed the current requests and historical requests to LLM via the request rewriting template to revise the current request

class Request_Rewriter():
    
    def __init__(self, model, tokenizer, kwargs) :
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
    
    def request_rewrite(self, history_rewrite_request, request) :
        history_text = "".join(history_rewrite_request)

        #If there is no history requests, do not rewrite query.
        if (len(history_rewrite_request) == 0) :
            return request

        else :
            query_rewrite_input = global_no_demon_template.format(input=request_rewriting_template.format(history = history_text, request = request))
            revised_request = generate_response(self.model, self.tokenizer, query_rewrite_input, **self.kwargs)
            
            #show the rewritten query on the demo to verficate whether the rewriting is correct.
            st.write("---")
            st.write("the rewritten request is:" + revised_request)

        return revised_request
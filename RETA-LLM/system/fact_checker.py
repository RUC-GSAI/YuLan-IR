from config import *
from model_response import generate_response
# This is the fact checker module. 
# We feed the reference and answer to the LLM via fact check template to verify if there exist factual mistakes in the answer.
# TBD

class Fact_Checker():
    def __init__(self, model, tokenizer) :
        self.model = model
        self.tokenizer = tokenizer
    
    def fact_check(self, reference, answer):
        ### TBD
        ### Following is a priliminary attempt for fact checking, which is problematic. So we simply return True for fact cheking
        ### We will soon release a useful version for this

        #answer_input = fact_checking_template.format(context = reference, answer = answer)
        #input_text = global_no_demon_template.format(input=answer_input)
        #output = generate_response(self.model, self.tokenizer, input_text, **kwargs)
        #if ("no" in output or "No" in ouput or "NO" in output) :
        #    return True
        #if ("yes" in output or "Yes" in ouput or "YES" in output) :
        #    return False
        return True
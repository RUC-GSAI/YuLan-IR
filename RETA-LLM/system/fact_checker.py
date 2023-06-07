from config import *
from model_response import generate_response
# This is the fact checker module. 
# We feed the reference and answer to the LLM via fact check template to verify if there exist factual mistakes in the answer.
# TBD

class Fact_Checker():
    def __init__(self, model, tokenizer, kwargs) :
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
    
    def fact_check(self, reference, answer):
        return True
        ### TBD
        ### Following is a priliminary attempt for fact checking, which have not been sufficiently tested and verified. 
        ### So we simply return True for fact cheking
        ### If you do want to use the following code, we recommend to use YuLan-65B as LLM backbone.
        ### We will soon release a more mature version for this

        answer_input = fact_checking_template.format(reference = reference, answer = answer)
        input_text = global_no_demon_template.format(input=answer_input)
        output = generate_response(self.model, self.tokenizer, input_text, **self.kwargs)
        if ("no" in output or "No" in output or "NO" in output) :
           return True
        if ("yes" in output or "Yes" in output or "YES" in output) :
           return False
        return True


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
        #TBD
        return True
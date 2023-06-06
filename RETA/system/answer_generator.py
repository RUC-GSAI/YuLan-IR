from config import *
from model_response import generate_response
# This is the answer generator module. 
# We feed the reference and question to LLM via the answer generation template to generate the final answer.

class Answer_Generator():
    def __init__(self, model, tokenizer) :
        self.model = model
        self.tokenizer = tokenizer
    
    def answer_generate(self, reference, question):
        answer_input = answer_generation_template.format(context = reference, question = question)
        input_text = global_no_demon_template.format(input=answer_input)

        output = generate_response(self.model, self.tokenizer, input_text, **kwargs)

        return output
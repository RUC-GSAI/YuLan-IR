from config import *
from model_response import generate_response
# This is the passage extractor module. 
# We feed the document contents and question to LLM via the passage extraction template to extract the relevant passages or fragments from the documents.

class Passage_Extractor():
    
    def __init__(self, model, tokenizer, kwargs) :
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.extract_step = extract_step
        self.extract_window = extract_window
    
    def extract(self, raw_reference, query, if_extract = True) :
        
        if (if_extract) :
            ref_length = len(raw_reference["contents"])

            passage = ""

            for j in range(0, ref_length, extract_step) :             
                extraction_input = passage_extraction_template.format(content = raw_reference["title"] + ":" + raw_reference["contents"][j : j + extract_window], question=query)
                real_extraction_input = global_template.format(demon1 = demon1, summary1 = summary1, demon2 = demon2, summary2 = summary2, input = extraction_input)
                fragments = generate_response(self.model, self.tokenizer, real_extraction_input, **self.kwargs)
                passage = passage + fragments + "\t"
        else :
            passage = raw_reference["contents"]
        
        return passage
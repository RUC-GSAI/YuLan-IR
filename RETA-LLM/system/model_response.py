import torch
import openai
from config import openai_api_key
# This file is used to generate response from LLM.
# Below is the universal generation function for Huggingface transformers LLMs
# Edit this to let your own LLM (api) can response.
@torch.no_grad()
def generate_response(model, tokenizer, input_text, **kwargs):
    if model == "chatgpt":
        openai.api_key = openai_api_key
        response_text = chatgpt_response(input_text)
    else:
        inputs = tokenizer([input_text], padding=True, return_tensors='pt')
        new_input_text = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        device = next(iter(model.parameters())).device
        input_ids = inputs['input_ids'].to(device)
        outputs = model.generate(input_ids, **kwargs)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print(output_text)
        response_text = output_text[0][len(new_input_text[0]):].strip()
        #print(response_text)
        del input_ids
    return response_text

def chatgpt_response(input_text):
    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": input_text}]
            )
    response_text = completion["choices"][0]['message']['content']
    return response_text
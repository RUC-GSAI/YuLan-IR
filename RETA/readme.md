# RETA-LLM

A RETrieval-Augmented Large Language Model Toolkit

## Background

Large Language Models (LLMs) have shown extraordinary abilities in many areas. However, studies have shown that they still tend to hallucinate and generate responses opposite to facts sometimes. To solve these problems, researchers propose a new paradigm to strengthen LLMs with information retrieval systems (Retrieval-augmented LLMs), which enable LLMs to look up relevant contents from external IR system. Furthermore, by enhancing in-domain data resources, Retrieval-augmented LLMs can answer in-domain questions such as "Who is the dean of the Gaoling School of Artificial Intelligence of Renmin University of China?"

## Introduction

To support research in this area and help users build their own in-domain QA system, we devise **RETA-LLM**, a **RET**reival-**A**ugmented LLM toolkit. Compared with previous LLM toolkits such as Langchains, our RETA tookit focuses on retrieval-augmented LLMs and provides more optional plug-in modules. We also disentangles the LLMs and IR system more entirely, which makes you can customize search engines and LLMs.

The overall framework of our toolkit is shown as follows: ![RETA-LLM Framework](./resource/framework.png)

In general, there includes five steps/modules in our RETA-LLM tookit. 

- **Request Rewriting**: First, RETA-LLM utilizes LLMs to revise current request of users based on their histories to make it complete and clear. 
- **Doc Retrieval**: Second, RETA-LLM uses the revised user request to retrieve relevant documents from customized document corpus. In our demo, We use [disentangled-retriever](https://github.com/jingtaozhan/disentangled-retriever) as retriever. you can customize your own searcher.
- **Passage Extraction**: Third, since concatenating the whole relevant document content may be too long for LLMs to generate responses, RETA-LLM extracts relevance document fragments/passages by LLMs from the retrieved documents to form references for generation.
- **Answer Generation**: Fourth, RETA-LLM provides the revised user request and references for LLM to generate answers.
- **Fact checking**: Finally, RETA-LLM applies LLMs to verify whether the generate answers contain factual mistakes and output final responses for user request.

## Requirements
The requirements of our RETA-LLM toolkit is wrapped in the `environment.yml` file, install them by :

```
 conda env create -f environment.yml
 conda activate retallm
```


## Usage

We provide a complete pipeline to help you use your own customized materials (e.g. html files crawled from websites) to build your own RETA-LLM toolkit. The pipeline is as follows:

0. Prepare your html file resouces in the `raw_data` folder and the mapping table file `url.txt` that maps the websited urls to the filename;  The `url.txt` should be in a tsv format. The format of each line is:

   `file name(without ".html") \t url`

   We give example data and url_file in `sample_data.zip` and `sample_url.txt`. 
   Follow the usage guidelines, you can build a RUC-admission-assistant using them.

1. run the `html2json.py` in the `html2json` folder to convert html resources to json files.
   ```
   cd html2json
   python html2json.py --input_dir ../raw_data --output_dir ../json_data --url_file ../url.txt
   python deduplication.py # This code is used to remove duplicated n-grams in the processed json files among all html files. 
   cd ..
   ```
   The `json_data` is the ouput data directory containing json files.
   
2. run the `index_pipeline.py` in the `indexer` folder to build faiss-supported index and conduct domain-adaption.
   ```
   cd indexer
   python index_pipeline.py --index_type all  --data_dir ../json_data  --index_save_dir ../index --batch_size 128 --use_content_type title --train_dam_flag
   cd ..
   ```
   The `index` is the faiss-supported index directory. The args `--use_content_type` is used to indicate which parts (title, contents, all) of the documents is to used to build indexes. We suggest to conduct domain adaption. If you choose not to, remove the `--train_dam_flag` args and change the `DAM_NAME` config in `./system/config.py` folder. 

3. Prepare an LLM and its generating configuration json file in the `system` folder. An example json for LLama and ChatGLM is shown in `system/llm_llama.json` and `system/llm_chatglm.json`. The genrating configuration mainly include the model path, temperature, top_p, top_k, etc. Specifically, you can even use differnt LLMs in different modules.

4. run the `web_demo.py` in `system` folder to start serving. 
   ```
   cd system
   streamlit run web_demo.py --server.port 1241
   ```
   Then you can try your own RETA-LLM toolkit on server.ip:1241 !
   
   The configuration of the `web_demo.py` is in the `config.py` in `system` folder.  Please adjust the configuration if you use your own data. 
   
   For the LLMs, we provide the model loading and response template for LLama and ChatGLM in `load_model.py` and `model_response.py` in `system` folder, If you want to use other LLMs, please adjust these two files.

   For the searchers, we define a template for your customized searcher, see it in the `Common_Searcher` class in the `./system/searcher.py`.



## Acknowledgements
Thanks Jingtao for the great implementation of [disentangled-retriever](https://github.com/jingtaozhan/disentangled-retriever).


## License
RETA-LLM uses MIT License. All data and code in this project can only be used for academic purposes.
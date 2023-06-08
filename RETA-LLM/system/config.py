import json
import streamlit as st

#instruction template
global_template_yulan = "The following is a conversation between a human and an AI assistant namely YuLan, developed by GSAI, Renmin University of China. The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n[|Human|]:{demon1}\n[|AI|]:{summary1}\n[|Human|]:{demon2}\n[|AI|]:{summary2}\n[|Human|]:{input}\n[|AI|]:" #global instructions with two demonstrations
global_no_demon_template_yulan = "The following is a conversation between a human and an AI assistant namely YuLan, developed by GSAI, Renmin University of China.The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n[|Human|]:{input}\n[|AI|]:"#global instructions without demonstrations 
global_template_other = "[|Human|]:{demon1}\n[|AI|]:{summary1}\n[|Human|]:{demon2}\n[|AI|]:{summary2}\n[|Human|]:{input}\n[|AI|]:" #global instructions with two demonstrations 
global_no_demon_template_other = "{input}"#global instructions without demonstrations 


global_template = global_template_yulan
global_no_demon_template = global_no_demon_template_yulan

answer_generation_template_en = 'Now, entirely based on the following reference, please answer the query more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The query is delimited by triple parentheses ((())). You are not allowed to add fabrications to your answers. Please answer in Chinese. Reference: [[[{context}]]], query: ((({question}))), answer:'#answer generation instruction 
answer_generation_template_zh = '现在，请完全的基于用书名号分割《》的参考信息，简洁和专业的来回答引号“”分割的用户的问题。不允许在答案中添加编造成分，答案请使用中文。参考信息:《{context}》 问题:“{question}”' #answer generation instruction
answer_generation_template = answer_generation_template_en

request_rewriting_template_en = 'Your job now, as the senior psychologist, is to understand what the user is really trying to say. In the course of a conversation with you, the user makes a series of requests that are coherent, but the requests themselves may not be complete and clear because the subject or pronoun or other parts of one request may be missing and appear in the previous requests. Therefore, it is now up to you to complete the user current question request by understanding historical request. The user historical requests were :{history}, and the user current request is : {request}. Please rewrite the user current request so that its meaning is complete and clear. Finally, only a rewritten request is printed, without any additional explanation or intermediate process. Please output in Chinese. The rewritten request is:'#passage extraction instructions
request_rewriting_template_zh = '你现在作为资深心理学家，你的工作是理解用户的真实意图。在与你谈话的过程中，用户提出了一系列问题请求，这些问题请求是连贯的，但是这些问题请求本身可能并不完整，因为某个问题请求的主语或代词或其他部分可能是缺失的，出现在上一个问题中。因此，现在需要你通过理解上文用户提出的问题请求，把用户现在的问题请求补充完整。用户的已经提出的问题请求是:{history}，现在用户的问题请求是:{request}。请对用户的问题请求进行改写，使其含义明确。最后仅输出一个改写后的问题请求，不需要任何其他解释和中间过程。改写后的问题请求是：'#passage extraction instructions
request_rewriting_template = request_rewriting_template_en


passage_extraction_template_en = "From the given document, please select and ouput the relevant document fragments which are related to the query. The document is delimited by triple brackets [[[]]]. The query is delimited by triple parentheses ((())). Note that the output must be fragments of the original document, not a summary of the document. If there is no fragment related to the query in the document, please output nothing. Document content: [[[{content}]]], query:((({question}))), relevant document fragments:"#passage extraction instruction 
passage_extraction_template_zh = "请你从用书名号《》分割的文档内容中，选出和引号“”分割的查询有关的文档内容片段。注意，必须是文档原文片段，不能够总结和生成新的内容。文档内容：《{content}》，查询：“{question}”，相关文档片段："#passage extraction instruction 
passage_extraction_template = passage_extraction_template_en

fact_checking_template_en = "Now, based on the following reference, check whether the answer includes factual mistakes. The reference is delimited by triple brackets [[[]]]. The answer is delimited by triple parentheses ((())). Please only ouput Yes or No. Yes for including factual mistakes. No for not including factual mistakes. Reference: [[[{reference}]]], answer: ((({answer}))), output:"#fact checking instruction 
fact_checking_template_zh = "现在，根据下面的参考，判断答案是否包括事实错误。参考用三括号[[[]]]分隔。答案由三括号((())分隔。请只输出是或否。参考:[[[{reference}]]],答案:((({answer}))),输出:"#fact checking instruction 
fact_checking_template = fact_checking_template_en


#model_config
model_config_path = "./llm_yulan_large.json" #config path for LLM
openai_api_key = "your-key" #api_key for chatgpt if you use chatgpt


#searcher_config
search_type = "dense" # "sparse" or "common"
cutoff = 1000 #cutoff for reference
topk = 3 #numbers of relevant docs
language = 'zh' #the language for retrieval('zh' or 'en')
#The adapter download url should be consistent with the rem url in index_pipeline.py
rem_dict = {"zh":"https://huggingface.co/jingtao/REM-bert_base-dense-distil-dureader/resolve/main/lora192-pa4.zip",
            "en":"https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip"}
REM_URL = rem_dict[language]
### index_pathr and data_dir should be consistent with the args in index_pipeline.py.
DENSE_INDEX_PATH = "../index/dense/faiss.index"
SPARSE_INDEX_PATH = "../index/sparse"
DOC_PATH = "../json_data"
DAM_NAME = "../index/dam_module"
#if you do not conduct domain adaption in index_pipeline, use the following DAM_NAME
#DAM_NAME = "jingtao/DAM-bert_base-mlm-dureader" #for zh
#DAM_NAME = "jingtao/DAM-bert_base-mlm-msmarco" #for en

#extractor_config
if_extract = True #Extract passages from docs or directly use the contents
extract_step = 256 #The sliding window step for extracting passages
extract_window = 512 #The sliding window size for extracting passages

#demonstrations for passage extraction
content1 = "信息学院开创国内用“信息”一词来命名系和学科专业的先河，是我国计算机学科数据库领域教学和研究的开创者，已建成了“以数据为中心”的产学研用一体化人才高地。建有第一个经济科学实验室；数据库及大数据研究团队在国内学术界一直处于领先地位，2018年获“国家科学技术进步二等奖”。学院计算机学科居全国第一方阵。学院现有5个本科专业，其中计算机科学与技术、信息管理与信息系统、数据科学与大数据技术、信息安全获评国家级一流本科专业建设点，软件工程专业获评北京市级一流本科专业建设点。学院实行无时点专业选择，学生根据个人兴趣和发展规划，自主选择专业课程，形成下列个性化方向：计算机理论与技术、系统结"
query1 = "信息学院有哪些专业"
summary1 = "学院现有5个本科专业，其中计算机科学与技术、信息管理与信息系统、数据科学与大数据技术、信息安全获评国家级一流本科专业建设点，软件工程专业获评北京市级一流本科专业建设点。"
demon1 = passage_extraction_template.format(content = content1, question = query1) 

content2 = "信息资源管理学院:文理融合的数字化人才培养开创者：拥有完整的数字资源学科实验平台，在全国几十家行业领军单位挂牌建立教学实践基地，形成了完备的数字化应用型人才实践体系。信息学院开创国内用“信息”一词来命名系和学科专业的先河，是我国计算机学科数据库领域教学和研究的开创者，已建成了“以数据为中心”的产学研用一体化人才高地。"
query2 = "信息资源管理学院有什么特色"
summary2 = "文理融合的数字化人才培养开创者：拥有完整的数字资源学科实验平台，在全国几十家行业领军单位挂牌建立教学实践基地，形成了完备的数字化应用型人才实践体系。"
demon2 = passage_extraction_template.format(content = content2, question = query2)

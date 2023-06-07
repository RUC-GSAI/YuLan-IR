<p align="center">
<img src="https://github.com/RUC-GSAI/YuLan-IR/blob/main/yulan.jpg" width="400px">
</p>

# YuLan-IR

YuLan-IR is part of YuLan, an open source LLM initiative proposed by Gaoling School of Artificial Intelligence, Renmin University of China. 

## Overview

In this repository, we hope to explore the combination between Information Retrieval (IR) and generative language models, including but not limited to: 
- Augmenting generation with information retrieval to alleviate the hallucination of language models.
- Improving information retrieval with language models. For example, using language models to help IR models determine whether a document / passage is useful.
- etc.

## Current Progress

### RETA-LLM
**RETA-LLM** is a **RET**reival-**A**ugmented LLM toolkit to support research in retrieval-augmented generation and to help users build their own in-down LLM-based systems. Find more about it in the [repo](https://github.com/RUC-GSAI/YuLan-IR/tree/main/RETA-LLM).

### WebBrain
**WebBrain** is a new benchmark for retrieval-augmented generation. We collect both the first passage of Wikipedia and the corresponding references. Given a user query, retrieval models are required to return relevant references, while generation methods should generate a response with the help of the references. The whole pipeline simulates the working process of New Bing, while all data are accessible. More details can be found in the [repo](https://github.com/RUC-GSAI/YuLan-IR/tree/main/WebBrain).

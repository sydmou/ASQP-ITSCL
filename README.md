
<h1 align="center">
An Instruction Tuning-Based Contrastive Learning Framework for Aspect Sentiment Quad Prediction with Implicit Aspects and Opinions</h1>

<div align="center">

![](https://img.shields.io/badge/Sentiment_Analysis-ACOSQE-red)
![](https://img.shields.io/badge/Model-Pending-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]([https://opensource.org/licenses/MIT](https://img.shields.io/github/license/sydmou/ASQP-ITSCL))


Repo for EMNLP 2024 Findings Paper [ITSCL: An Instruction Tuning-Based Contrastive Learning Framework for Aspect Sentiment Quad Prediction with Implicit Aspects and Opinions](https://aclanthology.org/2024.findings-emnlp.453/).

## ✨ Introduction 

ITSCL is an instruction tuning-based contrastive learning method for ASQP Prediction:

- ITSCL  **combination of elements**:


<p align="center">
    <img src="./images/fig1.png" width="100%">
</p>


## Results Datasets

<p align="center">
   <img src="./images/fig3.png" width="100%"/>
</p>


## ✨ Methods 

<p align="center">
    <img src="./images/fig2.png" width="100%">
</p>



## Results
The quadruple extraction performance of five different systems (including baseline and +UAUL) on the four datasets:


<p align="center">
 <img src="./images/fig4.png" width="100%"/>
</p>

We further investigate the ability of ITSCL under SNE:

<p align="center">
   <img src="./images/fig5.png" width="100%"/>
</p>


## Environment
    Python==3.8
    torch==1.11.0
    transformers==4.14.1
    pytorch_lightning==0.8.1
    numpy==1.21.2


## The Parameter settings: 
    T5-base（Optimal Epochs）==50 (Restaurant)
    T5-base（Optimal Epochs）==50 (Laptop)
    T5-large（Optimal Epochs）==30 (Restaurant)
    T5-large（Optimal Epochs）==35 (Laptop)  
 
    
 
## For the function def get_ITBPE_style_targets(sents, labels): in the data_utils.py file: 
    # Usage:
    # - The `prefix_sentenceRestaurantN` should be set according to the dataset or domain-specific requirements.
    #   For example:
    #   - In the "Restaurant" domain, use `prefix_sentenceRestaurantN`.
    #   - For the "Laptop" domain, use `prefix_sentenceLaptopN`.
    # - This provides flexibility to choose different prefix templates based on the dataset or analysis context


## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@inproceedings{zhang-etal-2024-instruction,
    title = "An Instruction Tuning-Based Contrastive Learning Framework for Aspect Sentiment Quad Prediction with Implicit Aspects and Opinions",
    author = "Zhang, Hao  and
      Cheah, Yu-N  and
      He, Congqing  and
      Yi, Feifan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.453",
    pages = "7698--7714",
}
```

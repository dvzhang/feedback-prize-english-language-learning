# Readme
> My contact info: [Twitter](https://twitter.com/dvzhangtz) [Linkedin](https://www.linkedin.com/in/tianzuo-zhang/) Wechat: dvzhangtz [Kaggle](https://www.kaggle.com/milesme)
> 
> I also upload my homework to [Github](https://github.com/dvzhang/feedback-prize-english-language-learning)


# 0. Background
### 0.1.	Goal:
Make an article scoring system for English Language Learners.
### 0.3.	Description:
As a Kaggle user ( [my account](https://www.kaggle.com/milesme) ), I found a [very interesting competition](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) . I really hope I can solve this problem in my homework.  
The goal of this competition is to assess the language proficiency of 8th-12th grade English Language Learners (ELLs). Utilizing a dataset of essays written by ELLs will help to develop proficiency models that better support all students.
This work will help ELLs receive more accurate feedback on their language development and expedite the grading cycle for teachers. These outcomes could enable ELLs to receive more appropriate learning tasks that will help them improve their English language proficiency.
### 0.3.	One must be available via external public API. (You should be able to access it without a ton of trouble)
The train set and test set of this competition can be easily downloaded using Kaggle API.
I can use them to fine-tune my pretrain language model to get a scoring model.
However, the pretrain language model, Bert , or other models based on the Transformer, trained on data like Wikipedia. It is not compatible with our task. Some study shows the benefit of continued pretraining on domain-specific unlabeled data   .  
### 0.4.	One must require “scraping” (i.e., not available via external API) 
I want to use Scrapy to crawl [the website Lang-8](https://lang-8.com/1) , which is a second-language learners’ website and has lots of second-language users’ blogs. Using these to continue pretrain my pretrain language will make it more compatible with our task.
### 0.5.	The third can be an API, scraped, or a database
We can find some [similar competition on Kaggle](https://www.kaggle.com/competitions/feedback-prize-2021)   . Both of them release their dataset in the past. They can also be the continue pretrain data. 
![diagram](pic/WechatIMG553.png)

# 1. Setup you env
```shell
conda create -n kaggle python=3.7
conda activate kaggle

pip install kaggle
pip install lxml

mkdir -p input/commonlitreadabilityprize
```

# 2. Run the code
### 2.1. scraper.py --scrape
```shell
python scraper.py --scrape
```
- This will scrape the data but return only 5 entries of each dataset.
![pic](pic/WechatIMG555.png)

### 2.2. scraper.py --static <path_to_dataset>
```shell
python scraper.py --static ./data/static/lang8.csv
```
- This will return the static dataset scraped from the web and stored in database or CSV ﬁle
![pic](pic/WechatIMG554.png)

### 2.3. scraper.py
```shell
scraper.py 
```
- Return the complete scraped datasets.
- Kindly remind: it is very very slow, since this website will block crawler's IP, and I did not use IP pool. So I sleep about half minute after I crawl every page. If you must run it, use tmux to keep it running.
![pic](pic/WechatIMG556.png)

# 3. Download Data From Api
## 3.1 Download this competition's data from Kaggle API
### 3.1.1 How to use Kaggle api
Firstly, log in to [Kaggle](https://www.kaggle.com/), go to ACCOUT
![pic](./pic/20180502212652888)
Secondly, you can create New Api, getting a kaggle.json file
![pic](./pic/20180502212721622)

Thirdly, copy this file to your home/.kaggle
For example, I copy it to my ~/.kaggle, since I use Ubuntu

### 3.1.2 How to download data from Api
Download is easy

```shell
cd input/commonlitreadabilityprize
kaggle competitions download -c feedback-prize-english-language-learning
kaggle competitions download -c commonlitreadabilityprize
unzip commonlitreadabilityprize.zip
```

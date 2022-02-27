# multilingual-translation
This code evaluates neural machine translation benchmarks for low-resource languages using COMET and BLEU metrics. The bilingual framework uses parallel data from one high-resource language English and two low-resource languages Azerbaijani and Belarus. Our multilingual model boosts the performance of low-resource languages by using parallel datasets from similar languages that are higher resourced. This framework supplements our Azerbaijani dataset with Turkish and Belarusian with Russian.<br>

This work was derived from ["When and Why are Pre-Trained Word Embeddings Useful for Neural Machine Translation?"](https://arxiv.org/pdf/1804.06323.pdf)

## Requirements
* GPU environment that can run CUDA. We used an AWS Deep Learning AMI GPU PyTorch 1.10.0 (Amazon Linux 2) 20211115 with g4dn.xlarge instance
* Approximately 100 GB volume or memory. [This tutorial](https://wszhan.github.io/2018/04/19/no-space-on-AWS-EC2.html#:~:text=From%20aws%20console%2C%20click%20on,your%20instance%2C%20choose%20Detach%20Volume%20.) is helpful if you require more memory on your AWS volume after you've trained your models. 

## Code 

### Environment Setup 
Initialize your conda environment 
```
conda create -n your_env python=3.8
conda activate your_env
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
This code requires the machine translation framework [fairseq](https://github.com/pytorch/fairseq) added to your environment<br>
```
git clone git@github.com:pytorch/fairseq.git
cd fairseq
pip install .
pip install --upgrade numpy
export FAIRSEQ_DIR=`pwd`
cd ..
```
Lastly, this code requires this repo to be cloned and requirements installed into your environment
```
git clone https://github.com/charityking2358/multilingual-translation.git
cd multilingual-translation
pip install -r requirements.txt
```
### Data Download
Our data consists of TED Talk data in 58 parallel languages to English. 
The raw data can be downloaded [directly](https://docs.google.com/uc?export=download&id=1L1v_wwa8GwEGUy39Xls1JF2VoU0hSDJ7) or by using the below command to run the download script.  
```
python download_data.py
```
### Bilingual Baselines<br>
These scripts train our neural machine translation systems only using parallel data from languages of interest. For our baseline, we will evaluate the low-resource languages Azerbaijani (aze) and Belarus (bel), translating to and from our high-resource language English (eng). 
<br> 

 1. Preprocessing<br>
 These bash scripts do simple data cleaning, subword tokenization, and data size partitioning for training aze-eng and bel-eng bidirectionally. 
 ```
 bash preprocess-ted-bilingual.sh
 bash preprocess-ted-bilingual-bel.sh
 ```
 2. Train and Evaluate
These scripts train and evaluate models from preprocessed data in both directionn from English and to English by running: 
 ```
bash traineval_aze_eng.sh
bash traineval_eng_aze.sh
bash traineval_bel_eng.sh
bash traineval_eng_bel.sh
```
Our benchmark results provide the following scoring metrics: 
Language Pair  |  BLEU  | COMET
------------- | -------------  | -------------
aze-eng  | 1.96 |  -1.1737
eng-aze  | 1.54 |  -1.3101
bel-eng  | 1.39 |  -1.3867
eng-bel  | 1.29 |  -1.3987

### Multilingual Baselines<br> 
For low-resource languages with scant parallel training data, our NMT models perform poorly with BLEU scores < 10 and negative COMET scores. A multilingual approach that boosts performance to low-resource languages uses training data from languages similar to our target language from higher resourced languages. For this example, we supplement Azerbaijani (aze) with Turkish (tur) and Belorussian (bel) with Russian (rus). We increase our training data by concatenating the low-resource and supplementary high resource languages together to train our model. 
1) Preprocess our languages<br>
These scripts preprocess our aze-tur language pair and bel-rus language pair. 
```
bash preprocess-ted-multilingual.sh
bash preprocess-ted-multilingual-bel.sh
```
2) Train and Evaluate
These scripts train and evaluate models from preprocessed data in both directionn from English and to English by running: 
```
bash traineval_azetur_eng.sh
bash traineval_eng_azetur.sh
bash traineval_belrus_eng.sh
bash traineval_eng_belrus.sh
```
  
Our benchmark results provide the following scoring metrics: 
Language Pair  |  BLEU  | COMET
------------- | -------------  | -------------
azetur-eng  | 11.97 |  -0.206
eng-azetur  | 6.05 |  -0.0913
belrus-eng  | 17.47 |  -0.3419
eng-belrus  | 9.91 |  -0.4414
  
  
  

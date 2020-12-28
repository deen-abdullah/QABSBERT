# **QABSBERT**

This code is for INLG 2020 paper  [Towards Generating Query to Perform Query Focused Abstractive Summarization using Pre-trained Model](https://www.aclweb.org/anthology/2020.inlg-1.11/)

Some codes are collected from the work of [Liu and Lapata](https://github.com/nlpyang/PreSumm)

We conducted our experiment on TITAN X GPU (GTX Machine) and later we have uploaded our work on GitHub for general use.
Experiment was run on Anaconda environment. Please install required packages.

Our summarization framework has two parts, at first we pre-processed the source document according to the query by which we incorporated the query relevance to our QFAS task. Then, we used the [BERTSUM](https://github.com/nlpyang/PreSumm) model to generate abstractive summaries, where we fine-tuned the model with our pre-processed source documents.

## **Pre-processing / Dataset Preparation**

### **CNN/DailyMail Dataset**

**Step 1 Download Stories:** Download and unzip the stories directories from [here](https://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all .story files in one directory (e.g. ../raw_stories)

**Step 2. Download Stanford CoreNLP:** We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
<pre><code>
export CLASSPATH=/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
</code></pre>
replacing /path/to/ with the path to where you saved the stanford-corenlp-full-2018-10-05 directory.

As an example:
<pre><code>
export CLASSPATH=/home/user/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
</code></pre>

**Step 3. Sentence Splitting and Tokenization:**
<pre><code>
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
</code></pre>
RAW_PATH is the directory containing story files (../raw_stories), TOKENIZED_PATH is the target directory to save the generated tokenized files (../merged_stories_tokenized)

As an example:
<pre><code>
python preprocess.py -mode tokenize -raw_path ../raw_stories -save_path ../merged_stories_tokenized
</code></pre>
**Step 4. Format to Simpler Json Files:**
<pre><code>
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path MAP_PATH
</code></pre>
RAW_PATH is the directory containing tokenized files (../merged_stories_tokenized), JSON_PATH is the target directory to save the generated json files (../json_data/cnndm), MAP_PATH is the directory containing the urls files (../urls)

As an example:
<pre><code>
python preprocess.py -mode format_to_lines -raw_path ../merged_stories_tokenized -save_path ../json_data/cnndm -n_cpus 1 -use_bert_basic_tokenizer false -map_path ../urls
</code></pre>
**Step 5. Format to PyTorch Files:**
<pre><code>
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log
</code></pre>
JSON_PATH is the directory containing json files (../json_data), BERT_DATA_PATH is the target directory to save the generated binary files (../bert_data)

As an example:
<pre><code>
python preprocess.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data  -lower -n_cpus 1 -log_file ../logs/preprocess.log
</code></pre>

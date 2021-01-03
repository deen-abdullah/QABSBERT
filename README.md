# **QABSBERT**

This code is for INLG 2020 paper  [Towards Generating Query to Perform Query Focused Abstractive Summarization using Pre-trained Model](https://www.aclweb.org/anthology/2020.inlg-1.11/)

Some codes are collected from the work of [Liu and Lapata](https://github.com/nlpyang/PreSumm)

We conducted our experiment on TITAN X GPU (GTX Machine) and later we have uploaded our work on GitHub for general use.
Experiment was run on Anaconda environment. Please install required packages.

Our summarization framework has two parts, at first we pre-processed the source document according to the query by which we incorporated the query relevance to our QFAS task. Then, we used the [BERTSUM](https://github.com/nlpyang/PreSumm) model to generate abstractive summaries, where we fine-tuned the model with our pre-processed source documents.

Create a directory, namely 'logs' in the same directory where you have downloaded QABSBERT.

## **Pre-processing / Dataset Preparation**
Prepare only one dataset at a time: CNN/DailyMail or Debatepedia or Abstractive(Newsroom) or Mixed (Newsroom) and complete the full experiment.

For another dataset, we have to download the project and do the steps from the beginning.

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

### **Newsroom Dataset (Abstractive/Mixed)**

**Step 1 Download Dataset:** Download and unzip the dataset from [here](https://summari.es). To download full dataset, fill up the form in their website, then they will send the link to your email to download the full dataset.

There are three parts in the dataset (dev.jsonl.gz;  test.jsonl.gz; train.jsonl.gz). Put them in one directory (e.g. ../raw_stories)

**Step 2. Download Stanford CoreNLP:** We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
<pre><code>
export CLASSPATH=/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
</code></pre>
replacing /path/to/ with the path to where you saved the stanford-corenlp-full-2018-10-05 directory.

As an example:
<pre><code>
export CLASSPATH=/home/user/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
</code></pre>

**Step 3. Execute Command:**

For Abstractive Dataset of Newsroom:
<pre><code>
python preprocessNewsroom.py abstractive
</code></pre>
For Mixed Dataset of Newsroom:
<pre><code>
python preprocessNewsroom.py mixed
</code></pre>

### **Debatepedia Dataset**

**Step 1 Download Dataset:** Download dataset from [here](https://github.com/PrekshaNema25/DiverstiyBasedAttentionMechanism/tree/master/data).

There are six files in the dataset (test_content, test_summary, train_content, train_summary, valid_content, valid_summary). Put them in one directory (e.g. ../raw_stories)

**Step 2. Download Stanford CoreNLP:** We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
<pre><code>
export CLASSPATH=/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
</code></pre>
replacing /path/to/ with the path to where you saved the stanford-corenlp-full-2018-10-05 directory.

As an example:
<pre><code>
export CLASSPATH=/home/user/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
</code></pre>

**Step 3. Execute Command:**

<pre><code>
python preprocessDebatepedia.py
</code></pre>

## **Model Training**
We followed [BERTSUM](https://github.com/nlpyang/PreSumm) for the model training.

First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use -visible_gpus -1, after downloading, you could kill the process and rerun the code with multi-GPUs.

Replace XYZ with {cnndm | newsroom | debatepedia}

Run the following command: (After downloading BERT model, kill the process)
<pre><code>
python train.py  -task abs -mode train -bert_data_path ../bert_data/XYZ -dec_dropout 0.2  -model_path ../models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1  -log_file ../logs/abs_bert_XYZ
</code></pre>

Finally run the following command:
<pre><code>
python train.py  -task abs -mode train -bert_data_path ../bert_data/XYZ -dec_dropout 0.2  -model_path ../models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0  -log_file ../logs/abs_bert_XYZ
</code></pre>

## **Model Evaluation**

<pre><code>
python train.py -task abs -mode validate -test_all -batch_size 3000 -test_batch_size 500 -bert_data_path ../bert_data/XYZ -log_file ../logs/val_abs_bert_XYZ -model_path ../models -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_XYZ
</code></pre>

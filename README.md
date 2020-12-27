# **QABSBERT**

This code is for INLG 2020 paper  [Towards Generating Query to Perform Query Focused Abstractive Summarization using Pre-trained Model](https://www.aclweb.org/anthology/2020.inlg-1.11/)

Some codes are collected from the work of [Liu and Lapata](https://github.com/nlpyang/PreSumm)

We conducted our experiment on TITAN X GPU (GTX Machine) and later we have uploaded our work on GitHub for general use.

Our summarization framework has two parts, at first we pre-processed the source document according to the query by which we incorporated the query relevance to our QFAS task. Then, we used the [BERTSUM](https://github.com/nlpyang/PreSumm) model to generate abstractive summaries, where we fine-tuned the model with our pre-processed source documents.

## **Pre-processing / Dataset Preparation**

### **CNN/DailyMail Dataset**

**Step 1 Download Stories:** Download and unzip the stories directories from [here](https://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all .story files in one directory (e.g. ../raw_stories)

**Step 2. Download Stanford CoreNLP:** We will need Stanford CoreNLP to tokenize the data. Download it here and unzip it. Then add the following command to your bash_profile:

<pre><code>
export CLASSPATH=/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
</code></pre>

replacing /path/to/ with the path to where you saved the stanford-corenlp-full-2018-10-05 directory.

As an example:
<pre><code>
export CLASSPATH=/home/shifaabdia/Deen/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
</code></pre>

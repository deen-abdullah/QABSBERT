'''
##### For Abstractive Summarization ####
call 
	orderingQuery (source, tgt) 
								function which will return 
	source, query, tgt, sorting_status

sample syntax:
				source, query, tgt, sorting_status = orderingQuery (source, tgt)

##### For Extractive then Abstractive Summarization ####
call 
	orderingQuery (source, tgt, extractive = True) 
								function which will return 
	source, query, tgt, sorting_status

sample syntax:
				source, query, tgt, sorting_status = orderingQuery (source, tgt, extractive = True)

#### 
sorting_status = True / False
False: No query is matched with document and summary; no sorting is done for source document: Summarizer will use random selection
True: We have query related with document and summary; sorting is done for source document: Summarizer will take sentences from the top of the source document
'''
			
'''
# Algorithm during model tuning
if words from query is not matching with document then go with random sentence selection
otherwise select the sentences one by one from the begining of the source which is already sorted according to query
'''


import re
import operator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

nlp = spacy.load("en_core_web_md")  # make sure to use larger model!

def removeSpecialCharacter_Sent(sent):
	cleanSent = []
	for token in sent:
		if (re.match('[a-zA-Z]+', token)):
			cleanSent.append(token.lower())
	return cleanSent

#purpose: removing specialcharacter and lower case conversion
#input: sentences
#return: cleanSentences
def removeSpecialCharacter(sentences):
	cleanSentences = []
	for sent in sentences:
		cleanSent = removeSpecialCharacter_Sent(sent)
		cleanSentences.append(cleanSent)
	return cleanSentences


def removeStopWords_Sent (sent):
	stop_words = set(stopwords.words('english'))
	numberOfStopWords = 0
	filteredSentence = []
	for token in sent: 
		if token not in stop_words: 
			filteredSentence.append(token)
		else:
			numberOfStopWords += 1
	return filteredSentence

#purpose: removing stpwords
#input: sentences
#return: filteredSentences, numberOfStopWords
def removeStopWords (sentences):
	filteredSentences = [] 
	for sent in sentences:
		filteredSentence = removeStopWords_Sent(sent)
		filteredSentences.append(filteredSentence)
	return filteredSentences

def lemmatize_Sent (sent):
	lemmatizer = WordNetLemmatizer()
	filteredSentence = []
	for token in sent:
		filteredSentence.append(lemmatizer.lemmatize(token))
	return filteredSentence

#purpose: applying lemmatization
#input: sentences
#return: filteredSentences
def lemmatize (sentences):
	filteredSentences = []
	for sent in sentences:
		filteredSentence = lemmatize_Sent(sent)
		filteredSentences.append(filteredSentence)
	return filteredSentences

def takeSecond(elem):
	return elem[1]

def sortedFrequency (txt):
	frequency = {}

	for t in txt:
		for word in t:
			count = frequency.get(word,0)
			frequency[word] = count + 1

	freq = operator.itemgetter(1)
	frequency = sorted(frequency.items(), reverse=True, key=freq)

	return frequency

def sent_similarity (q, s):
	count = 0
	for token1 in q:
		for token2 in s:
			if (token1 == token2):
				count = count + 1
	similarity = float(count) / float(len (q))
	return similarity

def orderingQuery (source, target, query_size = 10, extractive = False):
	# processing to generating query
	sorting_status = True
	src = removeSpecialCharacter(source)
	src = removeStopWords(src)
	src = lemmatize (src)

	tgt = removeSpecialCharacter(target)
	tgt = removeStopWords(tgt)
	tgt = lemmatize (tgt)

	summary = ""
	for t in tgt:
		for token in t:
			if token not in summary:
				summary = summary  + ' ' + token;

	document = ""
	for d in src:
		for token in d:
			if token not in document:
				document = document  + ' ' + token;

	src = nlp(document)
	tgt = nlp(summary)

	frequency = {}

	for t in tgt:
		sim = 0.0
		count = 0
		for s in src:
			sim = sim + s.similarity(t)
			count = count + 1
		if count != 0:
			sim = sim / count
		frequency[t.text] = sim
	
	freq = operator.itemgetter(1)
	frequency = sorted(frequency.items(), reverse=True, key=freq)

	query = ""
	i = 0

	for k, v in frequency:
		query = query + ' ' + k
		i = i + 1
		if i == query_size:
			break

	# Query has been generated

	# Document sorting process 
	qry = nlp(query)
	source_sorted = {}
	hash_table = {}

	i = 0
	for sent in source:
		src_sent = removeSpecialCharacter_Sent(sent)
		src_sent = removeStopWords_Sent(src_sent)
		src_sent = lemmatize_Sent(src_sent)

		doc = ""
		for token in src_sent:
			if token not in doc:
				doc = doc  + ' ' + token;

		d = nlp (doc)
		score = qry.similarity(d)

		source_sorted[i] = score
		hash_table[i] = sent
		i = i + 1

	freq = operator.itemgetter(1)
	source_sorted = sorted(source_sorted.items(), reverse=True, key=freq)

	if (extractive):
		source = []
		for k, v in source_sorted:
			if (v > 0.0):
				source.append(hash_table[k])
	else:
		source = []
		for k, v in source_sorted:
			source.append(hash_table[k])
	
	return source, query, target, sorting_status


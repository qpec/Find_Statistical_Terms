import os
import fitz
import re
import nltk
import pandas as pd
import sys
import numpy as np
import string
from nltk.stem import SnowballStemmer
import collections
import unidecode

path = "."

def get_a_z_index():
	path = "./Books/new.pdf"
	pdf = fitz.Document(path)

	txt = []
	for page in pdf:
		enumerate([txt.append(i) for i in re.findall(r'.+\w\n', page.getText()) if i not in txt])

	str_text = ""
	for i in txt:
		for x in i:
			str_text += str(x)

	str_text = unidecode.unidecode(str_text)
	str_text = re.sub(r'.*\d{2,}.*', '', str_text)
	str_text = re.sub(r'.+\..*', '', str_text)
	str_text = re.sub(r'.+\..*', '', str_text)
	str_text = re.sub(r'\n[a-z][a-z].+', "", str_text)
	str_text = re.sub(r'\n+', '\n', str_text)
	str_text = re.sub(r' +', ' ', str_text)
	str_text = re.sub(r'-', ' ', str_text)
	str_text = re.sub(r'\'', '', str_text)
	str_text = str_text.lower()
	lst_text = str_text.split("\n")
	add_spaces = [" "+str(i)+" " for i in lst_text]

	return add_spaces

#establish corpus path and spacy dependancies
def path_finder(path):
	files = os.listdir(path)
	paths = []
	for f in files:
		if f.endswith('.pdf'):
			path_to_file = os.path.join(path, f)
			paths.append(path_to_file)
	return paths

def extract_text_and_clean(path):
	text = ''
	pdf_object = fitz.Document(path)
	for page in pdf_object:
		txt = page.getText()
		text += " " + txt
	pdf_object.close()
	text = text.lower()
	text = unidecode.unidecode(text)
	text = re.sub(r'\-\n', '', text)
	text = re.sub(r'\n', ' ', text)
	text = re.sub(r'[^\w\.\'\- ]', ' ', text)
	text = re.sub(r'-', ' ', text)
	text = re.sub(r'\'', '', text)
	text = re.sub(r' +', ' ', text)
	return text

def vectorize_index_words():
	includers = get_a_z_index()
	vector = {}
	mirror_stem_vector = {}
	token_index = -1
	stemmer = SnowballStemmer("english")
	for i in range(0, len(includers)):
		if includers[i] not in vector:
			token_index += 1
			vector[includers[i]] = token_index
			stemmed_token = " ".join([stemmer.stem(i) for i in includers[i].split()])
			mirror_stem_vector[includers[i]] = " "+stemmed_token+" "
	return vector, mirror_stem_vector

def generate_list_phrases_all_articles(path):
	paths = path_finder(path)
	phrases_of_all_articles = []
	mirror_stem_phrases_of_all_articles = []
	stemmer = SnowballStemmer("english")
	for path in paths:
		text = extract_text_and_clean(path)
		phrases_of_this_article = nltk.sent_tokenize(text)
		new_phrases_of_this_article = []
		mirror_stem_new_phrases_of_this_article = []
		for sentence in phrases_of_this_article:
			new_sentence = re.sub(r'\.', '', str(sentence))
			new_sentence = re.sub(r' +', ' ', str(new_sentence))
			new_phrases_of_this_article.append(" " + new_sentence + " ")
			stemmed_sentence = " ".join([stemmer.stem(i) for i in sentence.split()])
			mirror_stem_new_phrases_of_this_article.append(stemmed_sentence)
		phrases_of_all_articles.append(new_phrases_of_this_article)
		mirror_stem_phrases_of_all_articles.append(mirror_stem_new_phrases_of_this_article)

	return phrases_of_all_articles, mirror_stem_phrases_of_all_articles

def vectorize_individual_word_tokens_and_define_weight(vector):
	individual_words_vector = {}
	index = 0
	for item in vector:
		tokens = item.split()
		for token in tokens:
			if token not in individual_words_vector:
				individual_words_vector[token] = index
				index += 1
	#counts the amount of times an individual word is in vector incl collac
	individual_words_counts = [0 for i in individual_words_vector]
	for item in vector:
		tokens = item.split()
		for token in tokens:
			individual_words_counts[individual_words_vector[token]] += 1
	#generates weight of individual original vector item
	item_sum_weight = [0 for i in vector]
	item_product_weight = [0 for i in vector]

	for item in vector:
		tokens = item.split()
		sum_item_weight = 0
		product_item_weight = 1
		for token in tokens:
			weight = individual_words_counts[individual_words_vector[token]]
			sum_item_weight += (1/(weight))
			product_item_weight *= (1/weight)
		item_sum_weight[vector[item]] = (sum_item_weight*(len(tokens)+1))
		item_product_weight[vector[item]] = product_item_weight

	return item_sum_weight, item_product_weight

def match_words_and_vector(path, show_sentence_match=False):
	vector, mirror_stem_vector = vectorize_index_words()
	phrases, mirror_stem_phrases = generate_list_phrases_all_articles(path)
	sum_weights, product_weights = vectorize_individual_word_tokens_and_define_weight(vector)
	titles = [i for i in vector]
	matched_cum_counts = [0 for i in titles]
	matched_bin_counts = [0 for i in titles]
	bin_prod_weight = [0 for i in titles]
	token_matched_results = ["" for i in titles]
	sentence_matched_results = ["" for i in titles]

	text_file = open("Output Vector.txt", "w")
	text_file.write(str(vector))
	text_file.close()
	text_file = open("Output Sentences.txt", "w")
	text_file.write(str(phrases))
	text_file.close()
	stemmer = SnowballStemmer("english")
	for a in range(0, len(phrases)):
		this_article_bin_counts = matched_bin_counts[:]
		for s in range(0, len(phrases[a])):
			for token, v in vector.items():
				return_token = token
				count_this_token = False
				if token in phrases[a][s]:
					count_this_token = True
				else:
					if mirror_stem_vector[token] in mirror_stem_phrases[a][s]:
						return_token = mirror_stem_vector[token]
						count_this_token = True
				if count_this_token == True:
					token_matched_results[v] += (" | " + return_token)
					sentence_matched_results[v] += ("-" + phrases[a][s] + "\n")
					matched_cum_counts[v] += 1
					matched_bin_counts[v] = this_article_bin_counts[v] + 1

	for i in range(0, len(matched_bin_counts)):
		bin_prod_weight[i] = matched_bin_counts[i]*sum_weights[i]

	results_dict = {
	"Title": titles,
	"Total": matched_cum_counts,
	"Bin": matched_bin_counts,
	"Sum Weight": sum_weights,
	"Product Weights": bin_prod_weight,

	"Token Matches": token_matched_results,
	}

	if show_sentence_match == True:
		results_dict["Sentence Matches"] = sentence_matched_results

	df = pd.DataFrame(results_dict)

	return df

#match_words_and_vector(path, includers).to_excel("datasheet1.xlsx")
match_words_and_vector(path, show_sentence_match=False).to_excel("datasheet2.xlsx")
#match_words_and_vector(path, includers, stem=True, sentence_match=True)

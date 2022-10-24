import spacy
import nltk
import utils as utils

def load_corpus(tokenized=False):
	with open('alicia.txt', 'r') as file:
		text = file.readlines()

	corpus = [nltk.word_tokenize(linia) for linia in text] if tokenized else [fila.strip() for fila in text]
	return corpus

def main():
	corpus = load_corpus()

	nlp = spacy.load("es_core_news_sm")
	
	for i in range(len(corpus)):
		frase = corpus[i]
		res = nlp(frase)
		out = ""
		for token in res:
			out += token.text + "/" + token.pos_ + " "
		print(out)
main()

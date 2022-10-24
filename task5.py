import spacy
import nltk
import utils as utils

def load_corpus(tokenized=False):
	with open('alicia.txt', 'r') as file:
		text = file.readlines()

	corpus = [nltk.word_tokenize(linia) for linia in text] if tokenized else [fila.strip() for fila in text]
	return corpus

def tag(corpus, model):
	for i in range(len(corpus)):
		frase = corpus[i]
		res = model(frase)
		out = ""
		for token in res:
			out += token.text + "/" + token.pos_ + " "
		print(out)

def main():
	corpus = load_corpus()

	print("Small model")
	nlp = spacy.load("es_core_news_sm")
	print("Pipeline : " + str(nlp.pipe_names))
	tag(corpus,nlp)
	print()
	print("medium model")
	print("Pipeline : " + str(nlp.pipe_names))
	nlp = spacy.load("es_core_news_md")
	tag(corpus,nlp)
	print()
	print("large model")
	print("Pipeline : " + str(nlp.pipe_names))
	nlp = spacy.load("es_core_news_lg")
	tag(corpus,nlp)
	print()
	print("Transformer model")
	print("Pipeline : " + str(nlp.pipe_names))
	nlp = spacy.load("es_dep_news_trf")
	tag(corpus,nlp)
	
	
main()

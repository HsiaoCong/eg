#coding:utf-8

from nltk.tokenize.punkt import PunktSentenceTokenizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import networkx as nx
import numpy as np


document = """To Sherlock Holmes she is always the woman. I have
seldom heard him mention her under any other name. In his eyes she
eclipses and predominates the whole of her sex. It was not that he
felt any emotion akin to love for Irene Adler. All emotions, and that
one particularly, were abhorrent to his cold, precise but admirably
balanced mind. He was, I take it, the most perfect reasoning and
observing machine that the world has seen, but as a lover he would
have placed himself in a false position. He never spoke of the softer
passions, save with a gibe and a sneer. They were admirable things for
the observer-excellent for drawing the veil from men’s motives and
actions. But for the trained reasoner to admit such intrusions into
his own delicate and finely adjusted temperament was to introduce a
distracting factor which might throw a doubt upon all his mental
results. Grit in a sensitive instrument, or a crack in one of his own
high-power lenses, would not be more disturbing than a strong emotion
in a nature such as his. And yet there was but one woman to him, and
that woman was the late Irene Adler, of dubious and questionable
memory.
"""


# Sentence Spliting
document = ' '.join(document.strip().split('\n'))

sentence_tokenizer=PunktSentenceTokenizer()
sentences=sentence_tokenizer.tokenize(document)
#print (type(sentences))
for sent in sentences:
    print (sent)

# Bag of words
def bag_of_words(sentence):
    return Counter(word.lower().strip('.,') for word in sentence.split(' '))

# Tf-idf
c=CountVectorizer(ngram_range=(1,1),analyzer='word')
bow_array=c.fit_transform(sentences)

#获取词袋模型中所有词语
all_words=c.get_feature_names()
print (all_words)

#index2feature
#索引：词语
index2words={v:k for k,v in c.vocabulary_.items()}
print (index2words)

#m1为第一句话、第二句话词频
#m2为第三句话、第四句话词频
m1=c.transform(sentences[:2])
m2=c.transform(sentences[2:4])




#print (bow_array.toarray().shape)
#print (c.transform([sentences[0]]).sum(axis=0))
#print (c.vocabulary_)
#print (c.stop_words_)


#Converting to a Graph
normalized_matrix=TfidfTransformer(use_idf=True).fit(bow_array)
#print (normalized_matrix.toarray())
#m1_tf为第一句话、第二句话tf-idf
#m2_tf为第三句话、第四句话tf-idf

m1_tf=normalized_matrix.transform(m1)
m2_tf=normalized_matrix.transform(m2)

#print (m1.toarray().argsort()[:,-3:])
#print (m1.toarray().argsort())
print (np.argsort(-m1_tf.todense())[:,:3])
top_n_feature=(np.argsort(m1_tf.todense())[:,-3:])
print (top_n_feature)
print (all_words[46])

#求出top-n的关键词
print ((np.vectorize(index2words.get)(top_n_feature)))
"""
print("Nor")
print(TfidfTransformer(use_idf=False).fit_transform(bow_array).toarray())

#sentence similarity
similarity_graph=normalized_matrix*normalized_matrix.T
print (similarity_graph.shape)


# use pagerank to socre graph of sentences
nx_graph=nx.from_scipy_sparse_matrix(similarity_graph)
scores=nx.pagerank(nx_graph)
print (scores)

#mapping of sentence indices to scores
ranked=sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
print (ranked[0])
"""
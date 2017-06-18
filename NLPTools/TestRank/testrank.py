#coding:utf-8

#link :  http://joshbohde.com/blog/document-summarization

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
import numpy as np
import networkx as nx
import textrank



def TextRank(document):

    #分句
    sentence_tokenizer=PunktSentenceTokenizer()
    sentences=sentence_tokenizer.tokenize(document)

    #计算词频
    c=CountVectorizer()

    #计算tf-idf
    bow_matrix=c.fit_transform(sentences)
    normalized=TfidfTransformer().fit_transform(bow_matrix)


    #获取词袋模型中所有词语
    all_words=(c.get_feature_names())

    #index2word
    index2words = {v: k for k, v in c.vocabulary_.items()}

    #根据tf-idf 输出前三个关键字索引
    top_n_idx = np.argsort(normalized.todense())[:, -3:]
    #print(top_n_idx)

    #根据tf-idf 获取top-n关键字
    top_n_words=np.vectorize(index2words.get)(top_n_idx)
    #print (top_n_words)

    #计算文本相似性
    similarity_graph=normalized*normalized.T

    #构建图计算 textrank
    nx_graph=nx.from_scipy_sparse_matrix(similarity_graph)
    scores=nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)

if __name__=='__main__':
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
    TextRank(document)

    print (textrank.extractKeyphrases(document))
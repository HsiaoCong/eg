#coding:utf-8


from nltk.tokenize.punkt import PunktSentenceTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.decomposition import NMF,LatentDirichletAllocation

#nmf= 权重矩阵 * 特征矩阵
def nmf(document):

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

    nmf=NMF(n_components=2,random_state=27,alpha=0.1,l1_ratio=0.5).fit(normalized)

    # 权重矩阵
    w = nmf.transform(normalized)

    #特征矩阵
    f=nmf.components_.shape





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
    nmf(document)
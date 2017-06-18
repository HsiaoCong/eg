#coding:utf-8

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk
import re

def d2v(docuemnt):

    sentence_tokenizer=PunktSentenceTokenizer()
    document=re.sub('\n',' ',docuemnt)
    document=re.sub('\s+',' ',document)
    sentences=sentence_tokenizer.tokenize(document)
    print (len(sentences))
    prec_sentences=map(nltk.word_tokenize,sentences)
    labelized=[]
    for i,prec_sentences in enumerate(prec_sentences):
        labelized.append(TaggedDocument(prec_sentences,[str(i)+'_label']))
    print (labelized)

    model_dm = Doc2Vec(min_count=1, window=2, size=100, sample=1e-3,
                       workers=3)
    model_dm.build_vocab(labelized)
    print (np.array(model_dm.docvecs))
    print (np.array(model_dm.docvecs).shape)


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
    d2v(document)
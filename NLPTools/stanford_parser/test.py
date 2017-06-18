#coding:utf-8


from nltk.tokenize.stanford_segmenter import StanfordSegmenter

#中文分词
segmenter=StanfordSegmenter(
    path_to_jar="/home/hsiao/Develops/nlp/stanford-segmenter-2015-04-20/stanford-segmenter-3.5.2.jar",
    path_to_slf4j="/home/hsiao/Develops/nlp/stanford-corenlp-full-2016-10-31/slf4j-api.jar",
    path_to_sihan_corpora_dict="/home/hsiao/Develops/nlp/stanford-segmenter-2015-04-20/data",
    path_to_model="/home/hsiao/Develops/nlp/stanford-segmenter-2015-04-20/data/pku.gz",
    path_to_dict="/home/hsiao/Develops/nlp/stanford-segmenter-2015-04-20/data/dict-chris6.ser.gz"

)
str="我在我在博客园开了一个博客。"
print (segmenter.segment(str))

#英文分词


from nltk.tokenize import StanfordTokenizer
tokenizer=StanfordTokenizer(path_to_jar=r"/home/hsiao/Develops/nlp/stanford-parser-full-2016-10-31/stanford-parser.jar")
sent = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
print (tokenizer.tokenize(sent))

#中文命名实体识别
from nltk.tag import StanfordNERTagger
chi_tagger=StanfordNERTagger(model_filename=r'/home/hsiao/Develops/nlp/stanford-ner-2016-10-31/classifiers/chinese.misc.distsim.crf.ser.gz'
                             ,path_to_jar=r'/home/hsiao/Develops/nlp/stanford-ner-2016-10-31/stanford-ner.jar')
print (chi_tagger.tag('四川省 成都 信息 工程 大学 我 在 博客 园 开 了 一个 博客 ， 我 的 博客 名叫 伏 草 惟 存 ， 写 了 一些 自然语言 处理 的 文章 。\r\n'.split()))




#英文命名实体识别
from nltk.tag import StanfordNERTagger
eng_tagger=StanfordNERTagger(model_filename=r'/home/hsiao/Develops/nlp/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz'
                             ,path_to_jar=r'/home/hsiao/Develops/nlp/stanford-ner-2016-10-31/stanford-ner.jar')
print (eng_tagger.tag('Rami Eid is studying at Stony Brook University in NY'.split()))



#英文词性标注
from nltk.tag import StanfordPOSTagger
eng_tagger=StanfordPOSTagger(model_filename='/home/hsiao/Develops/nlp/stanford-postagger-full-2016-10-31/models/english-bidirectional-distsim.tagger',
                             path_to_jar='/home/hsiao/Develops/nlp/stanford-postagger-full-2016-10-31/stanford-postagger.jar')
print(eng_tagger.tag('What is the airspeed of an unladen swallow ?'.split()))


#中文词性标注
chi_tagger=StanfordPOSTagger(model_filename='/home/hsiao/Develops/nlp/stanford-postagger-full-2016-10-31/models/chinese-distsim.tagger',
                             path_to_jar='/home/hsiao/Develops/nlp/stanford-postagger-full-2016-10-31/stanford-postagger.jar')
print(chi_tagger.tag('四川省 成都 信息 工程 大学 我 在 博客 园 开 了 一个 博客 ， 我 的 博客 名叫 伏 草 惟 存 ， 写 了 一些 自然语言 处理 的 文章 。'.split()))



#英文句法分析
#import os
#java_path='/usr/lib/jvm/jdk/jdk1.8.0_121'
#os.environ['JAVAHOME']=java_path
from nltk.internals import find_jars_within_path
from nltk.parse.stanford import StanfordParser
eng_parser=StanfordParser('/home/hsiao/Develops/nlp/stanford-parser-full-2015-04-20/stanford-parser.jar',
                          '/home/hsiao/Develops/nlp/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar',
                          '/home/hsiao/Develops/nlp/stanford-corenlp-full-2016-10-31/englishPCFG.ser.gz')
eng_parser.__classpath=tuple(find_jars_within_path('/home/hsiao/Develops/nlp/stanford-parser-full-2016-10-31/'))
print (list(eng_parser.parse("the quick brown fox jumps over the lazy dog".split())))


#英文依存句法分析
from nltk.parse.stanford import StanfordDependencyParser
eng_parser=StanfordDependencyParser('/home/hsiao/Develops/nlp/stanford-parser-full-2015-04-20/stanford-parser.jar',
                          '/home/hsiao/Develops/nlp/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar',
                          '/home/hsiao/Develops/nlp/stanford-corenlp-full-2016-10-31/englishPCFG.ser.gz')
res = list(eng_parser.parse("the quick brown fox jumps over the lazy dog".split()))
print (res[0])
for row in res[0].triples():
    print(row)


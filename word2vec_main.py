import json
import re
from string import punctuation
import nltk
from nltk.tokenize import WordPunctTokenizer
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def data_process():
    #使用nltk分词分句器
    sent_tokenizer = nltk.data.load('data/tokenizers/punkt/english.pickle')
    word_tokenizer = WordPunctTokenizer()

    sentences_all=[]
    with open("data/yelp_academic_dataset_review.json.txt",encoding="utf-8") as f:
        lines=f.readlines()
        for line in lines:
            review=json.loads(line)#每个评论包含多个句子
            sentences=sent_tokenizer.tokenize(review['text'])
            sentences_all.extend(sentences)#句子列表

    punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'#去掉标点符号
    with open("data/review_sentences.txt","w",encoding="utf-8") as f:
        for sentence in sentences_all:
            sentence=re.sub(r"[{}]+".format(punc), " ", sentence)
            f.write(sentence+"\n")


def train_word2vec_model(seg_data,model_save_path,vector_save_path):
    model = Word2Vec(LineSentence(seg_data), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    #size是词向量的维度，workers多进程计算，只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。min_count,频数阈值，大于等于1的保留
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(model_save_path)
    model.wv.save_word2vec_format(vector_save_path, binary=False)

def test_model(model_path):
    model = Word2Vec.load(model_path)
    print(model.most_similar("excellent"))#获取一个词的相近词
    print(model.similarity("excellent","pleasure"))#两个词语的余弦相似度

#data_process()
#train_word2vec_model("data/review_sentences.txt", "data/word2vec.model", "data/word2vec.vector")
test_model("data/word2vec.model")
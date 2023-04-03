import gensim
from gensim.models import Word2Vec
from conf import *


def get_sentences():
    word_list = []
    sentences = ''
    with open(dic_path) as f:
        line = f.readline()[:-1]  # city\n
        word_list.append(line)
        while line:
            line = f.readline()[:-1]  # city\n
            word_list.append(line)
    f.close()
    return word_list

def get_tokens():
    word_list = get_sentences()
    model = Word2Vec([word_list], vector_size=d_model, window=n_cls, min_count=1, workers=4)
    tokens = torch.tensor(model.wv[word_list[0]]).unsqueeze(0)
    for i in range(1, len(word_list)-1):
        tokens = torch.cat([tokens, torch.tensor(model.wv[word_list[i]]).unsqueeze(0)], dim=0)
    return tokens

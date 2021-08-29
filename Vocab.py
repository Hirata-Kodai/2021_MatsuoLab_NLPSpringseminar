# import MeCab
import fugashi
import unidic
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD = 0
UNK = 1
MIN_COUNT = 1
word2id = {PAD_TOKEN: PAD, UNK_TOKEN: UNK}


class Vocab(object):
    """語彙を管理するクラス"""
    def __init__(self, word2id={}):
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, corpus, min_count=1, min_length=1):
        """テキストから語彙を構築するメソッド
        :pram corpus: list of list of str, コーパスとなるテキスト
        :pram min_count: int
        """
        word_counter = {}  # 単語の出現回数をカウント
        for sentence in corpus:
            for word in sentence:
                if len(word) >= min_length:
                    word_counter[word] = word_counter.get(word, 0) + 1

        # min_count以上出現する単語のみを語彙に追加
        # 出現回数の多い順にIDを振る

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word
        # 語彙に含まれる単語の出現回数
        self.row_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}


# 単語リストをIDのリストに変換する関数
def sentence_to_ids(vocab, sentence):
    """
    :pram vocab: vocabオブジェクト
    :pram sentence: list of str
    :return ids : list of int
    """
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    return ids


# 日本語分を形態素に分割する関数
def tokenize(text):
    """
    :param text: str, 日本語文
    :return tokenized: list of str, トークナイズされたリスト
    """
    tagger = fugashi.Tagger(f'-d {unidic.DICDIR}')
    tokenized = tagger.parseToNodeList(text)
    tokenized = [word.surface for word in tokenized]
    # tokenized = []
    # while node:
    #     if node.surface != '':
    #         tokenized.append(node.surface)
    #     node = node.next
    return tokenized



# パディングを行う関数
def pad_seq(seq, max_length):
    """Paddingを行う関数

    :param seq: list of int, 単語のインデックスのリスト
    :param max_length: int, バッチ内の系列の最大長
    :return seq: list of int, 単語のインデックスのリスト
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq

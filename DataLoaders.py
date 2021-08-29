from Vocab import pad_seq
import torch


class DataLoader_CBOW(object):
    """CBOW用のデータローダー"""
    def __init__(self, text, device, batch_size=50, window=2):
        """
        :pram text: 教師データ list of lists of int
        :pram batch_size
        :pram window
        """
        self.text = text
        self.batch_size = batch_size
        self.window = window
        self.w_pointer = 0  # 単語単位のポインタ
        self.s_pointer = 0  # 文単位のポインタ
        self.n_sent = len(text)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        """
        :return batch_X: (batch_size, window*2)のテンソル
        :return batch_Y: (batch_size, 1)のテンソル
        """
        batch_X = []
        batch_Y = []

        while len(batch_X) < self.batch_size:
            sent = self.text[self.s_pointer]
            target = sent[self.w_pointer]
            start = max(0, self.w_pointer - self.window)
            one_x = sent[start:self.w_pointer] + sent[self.w_pointer + 1:self.w_pointer + self.window + 1]
            one_x = pad_seq(one_x, self.window * 2)

            batch_X.append(one_x)
            batch_Y.append(target)

            self.w_pointer += 1
            if self.w_pointer >= len(sent):
                self.w_pointer = 0
                self.s_pointer += 1

                if self.s_pointer >= self.n_sent:
                    self.s_pointer = 0
                    raise StopIteration

        batch_X = torch.tensor(batch_X, dtype=torch.long, device=self.device)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=self.device)
        return batch_X, batch_Y


class DataLoaderSG(object):
    """Skipgramのためのデータローダー"""
    def __init__(self, text, device, batch_size, window=3):
        """
        :param text: list of list of int, 単語をIDに変換したデータセット
        :param batch_size: int, ミニバッチのサイズ
        :param window: int, 周辺単語と入力単語の最大距離
        """
        self.text = text
        self.batch_size = batch_size
        self.window = window
        self.s_pointer = 0 # データセット上を走査する文単位のポインタ
        self.w_pointer = 0 # データセット上を走査する単語単位のポインタ
        self.max_s_pointer = len(text) # データセットに含まれる文の総数
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch_X = []
        batch_Y = []

        while len(batch_X) < self.batch_size:
            sen = self.text[self.s_pointer]

            # Skipgramでは入力が1単語
            word_X = sen[self.w_pointer]

            # 出力は周辺単語
            start = max(0, self.w_pointer - self.window)
            word_Y = sen[start:self.w_pointer] + \
                sen[self.w_pointer + 1:self.w_pointer + self.window + 1]
            word_Y = pad_seq(word_Y, self.window * 2)

            batch_X.append(word_X)
            batch_Y.append(word_Y)
            self.w_pointer += 1

            if self.w_pointer >= len(sen):
                self.w_pointer = 0
                self.s_pointer += 1
                if self.s_pointer >= self.max_s_pointer:
                    self.s_pointer = 0
                    raise StopIteration

        batch_X = torch.tensor(batch_X, dtype=torch.long, device=self.device)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=self.device)

        return batch_X, batch_Y

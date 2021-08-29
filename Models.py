import torch
from torch import nn
import torch.nn.functional as F


def compute_loss(model, input, optimizer=None, is_train=True):
    """lossを計算するための関数
    is_train=Trueならモデルをtrainモードに、
    is_train=Falseならモデルをevaluationモードに設定します
    :param model: 学習させるモデル
    :param input: モデルへの入力
    :param optimizer: optimizer
    :param is_train: bool, モデルtrainさせるか否か
    """
    model.train(is_train)

    # lossを計算します。
    loss = model(*input)

    if is_train:
        # .backward()を実行する前にmodelのparameterのgradientを全て0にセットします
        optimizer.zero_grad()
        # parameterのgradientを計算します。
        loss.backward()
        # parameterのgradientを用いてparameterを更新します。
        optimizer.step()

    return loss.item()


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        """
        :param vocab_size: int, 語彙の総数
        :param embedding_size: int, 単語埋め込みベクトルの次元
        """
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.emb = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)  # Embedding層の定義
        self.linear = nn.Linear(self.embedding_size, self.vocab_size, bias=False)  # 全結合層（バイアスなし）

    def forward(self, batch_X, batch_Y):
        """
        :pram batch_X: Tensor(dtype=torch.long), (batch_size, window*2)
        :pram batch_Y: Tensor(dtype=torch.long), (batch_size, 1)
        :return loss: CBOWのロス
        """

        emb_X = self.emb(batch_X) # (batch_size, window*2, embedding_size)
        sum_X = torch.sum(emb_X, dim=1)  # (batch_size, embedding_size)
        lin_X = self.linear(sum_X)  # (batch_size, vocab_size)
        log_prob_X = F.log_softmax(lin_X, dim=-1) # (batch_size, vocab_size)
        loss = F.nll_loss(log_prob_X, batch_Y)
        return loss


class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        '''
        :pram vocav_size : int, 語彙の総数
        :pram embedding_size : int, 単語埋め込みベクトルの次元
        '''
        super(Skipgram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.linear = nn.Linear(self.embedding_size, self.vocab_size)

        def forward(self, batch_X, batch_Y):
            '''
            :pram batch_X : torch.Tensor(dtype=torch.long), (batch_size, )
            :pram batch_Y : torch.Tensor(dtype=torch.long), (batch_size, window*2)
            :return loss : torch.Tensor(dtype=forch.float), Skipgramのloss
            '''
            emb_X = self.embedding(batch_X)  # (batch_size, embedding_size)
            lin_X = self.linear(emb_X)  # (batch_size, vocab_size)
            log_prob_X = F.log_softmax(lin_X, dim=-1)  # (batch_size, vocab_size)
            log_prob_X = torch.gather(log_prob_X, 1, batch_Y)  # (batch_X, window*2)
            log_prob_X = log_prob_X * (batch_Y != 0).float()  # padding(=0) 部分にマスク
            loss = log_prob_X.sum(1).mean().neg()
            return loss


class SGNS(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        '''
        :pram vocav_size : int, 語彙の総数
        :pram embedding_size : int, 単語埋め込みベクトルの次元
        '''
        super(Skipgram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.linear = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, batch_X, batch_Y, batch_N):
        '''
        :pram batch_X : torch.Tensor(dtype=torch.long), (batch_size, )
        :pram batch_Y : torch.Tensor(dtype=torch.long), (batch_size, window*2)
        :pram batch_N : torch.Tensor(dtype=torch.long), (batch_size, n_negative)
        :return loss : torch.Tensor(dtype=forch.float), Skipgramのloss
        '''
        emb_X = self.embedding(batch_X)  # (batch_size, embedding_size)
        emb_Y = self.embedding(batch_Y)  # (batch_size, window*2, embedding_size)
        emb_N = self.embedding(batch_N)  # (batch_size, n_negative, embedding_size)
        

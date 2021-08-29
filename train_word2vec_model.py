import argparse
from Models import CBOW, Skipgram
from Vocab import Vocab, tokenize, sentence_to_ids
import DataLoaders
import pickle
import torch
import torch.optim as optim
import time
import numpy as np


def load_data(path):
    """テキストファイルを読み込むための関数
    :param path: str, ファイルパス
    :return text: list of list of str, 各文がトークナイズされたテキスト
    """
    text = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            line = tokenize(line)
            text.append(line)
    return text


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_data', type=str, default='./data/kokoro.txt',
                    help='A Text file to create embedding vector')
    ap.add_argument('--model_type', type=str, default=None,
                    help='Select model type(CBOW/SG)')
    ap.add_argument('--negative_sampling', action='store_true',
                    help='Do negative sampling')
    ap.add_argument('--dim_embedding', type=int, default=256,
                    help='Dimension of embedding vector')
    ap.add_argument('--window', type=int, default=2,
                    help='Window width')
    ap.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
    ap.add_argument('--epochs', type=int, default=20,
                    help='num epochs')
    ap.add_argument('--n_batches', type=int, default=1000,
                    help='num batches')
    ap.add_argument('--min_count', type=int, default=5,
                    help='Min count to add to vocab')
    ap.add_argument('--vocab_file', type=str, default=None,
                    help='To use existing Vocab object(pickle)')
    ap.add_argument('--save', type=str, default=None,
                    help='Path to save model')
    args = ap.parse_args()

    assert ~args.negative_sampling, 'Not inplemented yet'
    assert args.save is not None, 'Detect path to save'

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    PAD = 0  # <PAD>のID
    UNK = 1  # <UNK>のID
    # 辞書の初期化
    word2id = {
        PAD_TOKEN: PAD,
        UNK_TOKEN: UNK,
    }
    text = load_data(args.train_data)
    if args.vocab_file:
        vocab = pickle.load(args.vocab_file)
    else:
        vocab = Vocab(word2id=word2id)
    vocab.build_vocab(text, min_count=args.min_count)
    vocab_size = len(vocab.word2id)
    print("語彙数:", vocab_size)
    id_text = [sentence_to_ids(vocab, sen) for sen in text]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == 'CBOW':
        model = CBOW(vocab_size, args.dim_embedding)
        dataloader = DataLoaders.DataLoader_CBOW(id_text, device,  batch_size=args.batch_size, window=args.window)
    elif args.model_type == 'SG':
        model = Skipgram(vocab_size, args.dim_embedding)
        dataloader = DataLoaders.DataLoaderSG(id_text, device,  batch_size=args.batch_size, window=args.window)
    else:
        print('Model type is not selected. So try to train CBOW.')
        model = CBOW(vocab_size, args.dim_embedding)
        dataloader = DataLoaders.DataLoader_CBOW(id_text, device,  batch_size=args.batch_size, window=args.window)
    optimizer = optim.Adam(model.parameters())

    # Do train
    start_at = time.time()

    for batch_id, (batch_X, batch_Y) in enumerate(dataloader):
        loss = compute_loss(model, (batch_X, batch_Y), optimizer=optimizer, is_train=True)
        if batch_id % 100 == 0:
            print("batch:{}, loss:{:.4f}".format(batch_id, loss))
        if batch_id >= args.n_batches:
            break

    end_at = time.time()
    print("Elapsed time: {:.2f} [sec]".format(end_at - start_at))
    torch.save(model, args.save)


if __name__ == '__main__':
    main()

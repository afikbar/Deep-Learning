import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import config

import array
import os
import zipfile

import numpy as np
import six
import torch
import torch.nn as nn
from six.moves.urllib.request import urlretrieve
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, words_list):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.text = TextProcessor(
            classes=words_list,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.0,
        )

        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=1024,
            glimpses=glimpses,
            drop=0.2, )

        self.classifier = Classifier(
            in_features=(glimpses * vision_features, question_features),
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5, )

    def forward(self, v, b, q, v_mask, q_mask, q_len):
        '''
        v: visual feature      [batch, num_obj, 2048]
        b: bounding box        [batch, num_obj, 4]
        q: question            [batch, max_q_len]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        answer: predict logits [batch, config.max_answers]
        '''
        q = self.text(q, list(q_len.data))  # [batch, 1024]
        if config.v_feat_norm:
            v = v / (v.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(v)  # [batch, num_obj, 2048]

        a = self.attention(v, q)  # [batch, 36, num_glimpse]
        v = apply_attention(v.transpose(1, 2), a)  # [batch, 2048 * num_glimpse]
        answer = self.classifier(v, q)

        return answer


class TextProcessor(nn.Module):
    def __init__(self, classes, embedding_features, lstm_features, drop=0.0, use_hidden=True, use_tanh=False,
                 only_embed=False):
        super(TextProcessor, self).__init__()
        self.use_hidden = use_hidden  # return last layer hidden, else return all the outputs for each words
        self.use_tanh = use_tanh
        self.only_embed = only_embed
        classes = list(classes)

        self.embed = nn.Embedding(len(classes) + 1, embedding_features, padding_idx=len(classes))
        weight_init = torch.from_numpy(np.load(config.qa_path + '/glove6b_init_300d.npy'))
        assert weight_init.shape == (len(classes), embedding_features)
        print('glove weight shape: ', weight_init.shape)
        self.embed.weight.data[:len(classes)] = weight_init
        print('word embed shape: ', self.embed.weight.shape)

        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh() if self.use_tanh else nn.Identity()

        self.lstm = nn.GRU(input_size=embedding_features,
                           hidden_size=lstm_features,
                           num_layers=1,
                           batch_first=not use_hidden, )

    def forward(self, q, q_len):
        embedded = self.embed(q)
        embedded = self.drop(embedded)
        embedded = self.tanh(embedded)
        if self.only_embed:
            return embedded

        self.lstm.flatten_parameters()
        if self.use_hidden:
            packed = pack_padded_sequence(embedded, q_len, batch_first=True)
            _, hid = self.lstm(packed)
            return hid.squeeze(0)
        else:
            out, _ = self.lstm(embedded)
            return out


def obj_edge_vectors(names, wv_type='glove.6B', wv_dir=config.qa_path, wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)
    failed_token = []
    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word (hopefully won't be a preposition
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            # print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                failed_token.append(token)
    if (len(failed_token) > 0):
        print('Num of failed tokens: ', len(failed_token))
        # print(failed_token)
    return vectors


URL = {
    'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
    'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
    'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
    'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
}


def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        return torch.load(fname_pt)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


class Classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin11 = FCNet(in_features[0], mid_features, activate='relu')
        self.lin12 = FCNet(in_features[1], mid_features, activate='relu')
        self.lin2 = FCNet(mid_features, mid_features, activate='relu')
        self.lin3 = FCNet(mid_features, out_features, drop=drop)

    def forward(self, v, q):
        # x = self.fusion(self.lin11(v), self.lin12(q))
        x = self.lin11(v) * self.lin12(q)
        x = self.lin2(x)
        x = self.lin3(x)
        return x


class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)

        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)

        x = self.lin(x)

        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu')  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features, activate='relu')
        self.lin = FCNet(mid_features, glimpses, drop=drop)

    def forward(self, v, q):
        """
        v = batch, num_obj, dim
        q = batch, dim
        """
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)

        x = v * q
        x = self.lin(x)  # batch, num_obj, glimps
        x = F.softmax(x, dim=1)
        return x


def apply_attention(input, attention):
    """
    input = batch, dim, num_obj
    attention = batch, num_obj, glimps
    """
    batch, dim, _ = input.shape
    _, _, glimps = attention.shape
    x = input @ attention  # batch, dim, glimps
    assert (x.shape[1] == dim)
    assert (x.shape[2] == glimps)
    return x.view(batch, -1)

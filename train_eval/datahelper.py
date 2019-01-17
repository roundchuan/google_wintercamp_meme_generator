import os
import json
import torch
from torchvision import transforms
from scipy.misc import imread, imresize

PAD = 0
BOS = 1
EOS = 2
UNK = 3

min_count = 1
word_map_file = os.path.join('data', 'word_map_file')
# min_length = 0
max_length = 22     # include [BOS] and [EOS]


class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {"[pad]": PAD, "[start]": BOS, "[end]": EOS, "[unk]": UNK}
        self.word2count = {}
        self.index2word = {PAD: "[pad]", BOS: "[start]", EOS: "[end]", UNK: "[unk]"}
        self.num_words = 4  # count <pad>, <start>, <end>, <unk>

    def addSentence(self, sentence):

        # sentence is string
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:  # 已经过滤低频词了，不再继续过滤
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print("keep words {} / {} = {:.4f}".format(len(keep_words), len(self.word2count),
                                                   len(keep_words) / len(self.word2count)))

        # reinitialize dictionaries
        self.word2index = {"[pad]": PAD, "[start]": BOS, "[end]": EOS, "[unk]": UNK}
        self.word2count = {}
        self.index2word = {PAD: "[pad]", BOS: "[start]", EOS: "[end]", UNK: "[unk]"}
        self.num_words = 4  # count <pad>, <start>, <end>, <unk>

        for word in keep_words:
            self.addWord(word)


def loadImages(image_id):
    """
    Load image, convert an image to tensor.
    :param image_id: image_id
    :return: image tensor, (3, 224, 224)
    """

    # transform images
    transform = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    im_file = os.path.join('total', image_id)
    img = imread(im_file, mode="RGB")
    img = imresize(img, (224, 224)) # (224, 224, 3)
    img = img.transpose((2, 0, 1))  # (3, 224, 224)
    img = img / 255.
    img = torch.FloatTensor(img)
    img = transform(img)
    return img


def indexesFromSentence(word_map, caption):
    """
    Caption is string
    """
    ids = [BOS]
    if len(caption) < max_length-1:
        ids = ids + [word_map[word] if word in word_map else UNK for word in caption] + [EOS]
        while len(ids) < max_length:
            ids.append(PAD)
    else:
        for i in range(max_length - 2):
            if caption[i] in word_map:
                ids.append(word_map[caption[i]])
            else:
                ids.append(UNK)
        ids.append(EOS)
    return ids


def write_voc():
    with open('./total/total.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = list()
    for line in lines:
        tmp = list()
        tmp.append(line.split(" ", 1)[0])
        caption = line.split(" ", 1)[1].replace(' ', '').replace('\n', '').replace('"', '')
        tmp.append(caption)
        data.append(tmp)

    train = data[:5400]
    valid = data[5400:]
    voc = Voc()
    for line in train:
        voc.addSentence(line[1])
    voc.trim(min_count)

    # Write word map to file
    with open(word_map_file, 'w', encoding='utf-8') as j:
        json.dump(voc.word2index, j)


if __name__ == '__main__':
    write_voc()

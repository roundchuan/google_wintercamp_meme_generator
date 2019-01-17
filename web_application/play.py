"""
测试单张图片输入的输出字幕
"""
import json
import torch
import torch.backends.cudnn as cudnn
from datahelper import loadImages, indexesFromSentence
from utils import *
import torch.nn.functional as F
from torchvision import transforms
from scipy.misc import imread, imresize


def beam_search(im_path):

    """
    Evaluation.
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    # Parameters
    word_map_file = os.path.join('data', 'word_map_file')
    checkpoint = 'BEST_checkpoint.pth.tar'
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Load model
    checkpoint = torch.load(checkpoint)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2index)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    index2word = {value: key for key, value in word_map.items()}
    vocab_size = len(word_map)


    # TODO: Batched Beam Search

    # For each image
    k = 5

    # transform images
    transform = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img = imread(im_path, mode="RGB")
    img = imresize(img, (224, 224))  # (224, 224, 3)
    img = img.transpose((2, 0, 1))  # (3, 224, 224)
    img = img / 255.
    img = torch.FloatTensor(img)
    image = transform(img).unsqueeze(0)    # (1, 3, 224, 224)

    # Move to GPU device, if available
    image = image.to(device)

    # Encode
    encoder_out = encoder(image)    # (1, 14, 14, 2048) encoded_image_size = 14
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)    # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['[start]']]] * k).to(device)   # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words # (k, 1)

    # Tensor to store top k sequences's scores; now they are just 0
    top_k_scores = torch.zeros(k, 1).to(device) # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1) # (s, embed_dim)

        awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))   # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c)) # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['[end]']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 30:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    hyp = [index2word[ind] for ind in seq
           if ind not in {word_map['[start]'], word_map['[end]'], word_map['[pad]']}]
    return hyp


if __name__ == "__main__":
    hyp = beam_search('./test.jpg')
    print(hyp)

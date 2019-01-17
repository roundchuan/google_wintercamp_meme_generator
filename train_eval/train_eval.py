import os
import json
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from random import shuffle
from models import Encoder, DecoderWithAttention
from utils import *
from datahelper import *
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Model parameters
emb_dim = 512   # original 512
attention_dim = 512 # original 512
decoder_dim = 512   # original 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120    # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0    # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
encoder_lr = 1e-4   # learning rate for encoder if fine-tuning
decoder_lr = 4e-4   # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.    # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0. # BLEU-4 score right now
print_freq = 8
fine_tune_encoder = False   # fine-tune encoder?
checkpoint = None   # path to checkpoint, None if none


def main():
    """
    Training.
    """
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, word_map

    # Load word map
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune=fine_tune_encoder)  # False
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Load data
    with open('./total/total.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = list()
    for line in lines:
        tmp = list()
        tmp.append(line.split(" ", 1)[0])
        caption = line.split(" ", 1)[1].replace(' ', '').replace('\n', '').replace('"', '')
        tmp.append(caption)
        data.append(tmp)

    train_pairs = data[:5400]
    valid_pairs = data[5400:]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_pairs=train_pairs,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        recent_bleu4 = valid(valid_pairs=valid_pairs,
                             encoder=encoder,
                             decoder=decoder)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_pairs, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Perform one epoch's training.
    """
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()  # forward prop + back prop time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # compute BLEU score
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    shuffle(train_pairs)
    for batch in range(len(train_pairs) // batch_size):
        images = torch.FloatTensor()
        captions = list()
        caplens = list()
        refs = list()
        for i in range(batch_size * batch, batch_size * (batch + 1)):
            images = torch.cat([images, loadImages(train_pairs[i][0]).unsqueeze(0)], dim=0)
            caption = train_pairs[i][1]     # str "我喜欢天晴"
            refs.append(list(caption))  # BLEU reference, ref1 = [word1, word2, ...,]
            caplens.append(len(caption)+2 if len(caption)+2 <= max_length else 22)
            captions.append(indexesFromSentence(word_map, train_pairs[i][1]))

        data_time.update(time.time() - start)

        images = images.to(device)
        captions = torch.LongTensor(captions).to(device)
        caplens = torch.LongTensor(caplens).to(device)

        # Forward propagation
        images = encoder(images)  # (batch_size, 14, 14, 2048)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(images, captions, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]  # (batch_size, max_length-1), remove <start>

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()    # (batch_size, batch_max_length, vocab_size)
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back propagation
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of numbers
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, batch, len(train_pairs) // batch_size,
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # References
        sort_ind = sort_ind.tolist()
        for ind in sort_ind:
            references.append([refs[ind]])

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()  # (batch_size, batch_max_length)
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads

        # prediction results index to word
        index2word = {value: key for key, value in word_map.items()}
        for pred in temp_preds:
            hypotheses.append([index2word[ind] for ind in pred])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),
                        smoothing_function=SmoothingFunction().method1)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),
                        smoothing_function=SmoothingFunction().method1)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=SmoothingFunction().method1)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=SmoothingFunction().method1)
    print("BLEU-1: {}\tBlEU-2: {}\tBLEU-3: {}\tBLEU-4: {}".format(bleu1, bleu2, bleu3, bleu4))


def valid(valid_pairs, encoder, decoder):
    """
    Perform one epoch's validation.
    """
    decoder.eval()
    encoder.eval()

    # Load word map (word2index)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    index2word = {value: key for key, value in word_map.items()}
    vocab_size = len(word_map)

    # TODO: Batched Beam Search

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for batch in range(len(valid_pairs)):
        # for batch in tqdm(range(len(test_pairs)), desc="EVALUATING AT BEAM SIZE " + str(beam_size)):
        k = 5   # beam size

        image = loadImages(valid_pairs[batch][0]).unsqueeze(0)  # (1, 3, 224, 224)
        # Move to GPU device, if available
        image = image.to(device)

        # Encode
        encoder_out = encoder(image)  # (1, 14, 14, 2048) encoded_image_size = 14
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['[start]']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences's scores; now they are just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

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

        # References
        ref = list(valid_pairs[batch][1])
        references.append([ref])

        # Hypotheses
        hyp = [index2word[ind] for ind in seq
               if ind not in {word_map['[start]'], word_map['[end]'], word_map['[pad]']}]
        hypotheses.append(hyp)

        assert len(references) == len(hypotheses)

        if batch % 100 == 0:
            print("target: {}".format(' '.join(ref)))
            print("prediction: {}".format(' '.join(hyp)))
            print("------------------------")

    # Calculate BLEU-4 scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),
                        smoothing_function=SmoothingFunction().method1)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),
                        smoothing_function=SmoothingFunction().method1)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=SmoothingFunction().method1)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=SmoothingFunction().method1)
    return bleu4


if __name__ == '__main__':
    main()

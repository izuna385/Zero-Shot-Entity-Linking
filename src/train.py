import pdb, time, math
from utils import experiment_logger, cuda_device_parser, dev_or_test_finallog, worlds_loader
from parameters import Params
from data_reader import WorldsReader
import torch
import copy
import numpy as np
from embeddings import EmbLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from encoders import Pooler_for_mention, Pooler_for_title_and_desc
from model import Biencoder, WrappedModel_for_entityencoding
import torch.optim as optim
from evaluator import Evaluate_one_world
from token_indexing import TokenIndexerReturner

# SEEDS are FIXED
torch.backends.cudnn.deterministic = True
seed = 777
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    print("===experiment starts===")
    exp_start_time = time.time()
    P = Params()
    opts = P.opts
    experiment_logdir = experiment_logger(args=opts)
    P.dump_params(experiment_dir=experiment_logdir)
    cuda_devices = cuda_device_parser(str_ids=opts.cuda_devices)
    TRAIN_WORLDS, DEV_WORLDS, TEST_WORLDS = worlds_loader(args=opts)

    vocab = Vocabulary()
    iterator_for_training_and_evaluating_mentions = BucketIterator(batch_size=opts.batch_size_for_train, sorting_keys=[('context', 'num_tokens')])
    iterator_for_training_and_evaluating_mentions.index_with(vocab)

    embloader = EmbLoader(args=opts)
    emb_mapper, emb_dim, textfieldEmbedder = embloader.emb_returner()
    tokenIndexing = TokenIndexerReturner(args=opts)
    global_tokenizer = tokenIndexing.berttokenizer_returner()
    global_toknIndexer = tokenIndexing.token_indexer_returner()

    mention_encoder = Pooler_for_mention(args=opts, word_embedder=textfieldEmbedder)
    entity_encoder = Pooler_for_title_and_desc(args=opts, word_embedder=textfieldEmbedder)
    model = Biencoder(args=opts, mention_encoder=mention_encoder, entity_encoder=entity_encoder, vocab=vocab)
    model = model.cuda()
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=opts.lr, eps=opts.epsilon,
                           weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), amsgrad=opts.amsgrad)

    for epoch in range(opts.num_epochs):
        oneep_train_start = time.time()
        for world_name in TRAIN_WORLDS:
            reader = WorldsReader(args=opts, world_name=world_name, token_indexers=global_toknIndexer, tokenizer=global_tokenizer)
            trains = reader.read('train')
            trainer = Trainer(model=model, optimizer=optimizer,
                              iterator=iterator_for_training_and_evaluating_mentions, train_dataset=trains,
                              # validation_dataset=devs,
                              cuda_device=cuda_devices, num_epochs=1
                              )
            trainer.train()
        oneep_train_end = time.time()
        print('epoch {0} train time'.format(epoch+1), oneep_train_end - oneep_train_start, 'sec')

    print('====training finished=======')

    with torch.no_grad():
        model.eval()
        print('===Evaluation starts===')
        entity_encoder_wrapping_model = WrappedModel_for_entityencoding(args=opts, entity_encoder=entity_encoder, vocab=vocab)
        entity_encoder_wrapping_model.eval()

        for dev_or_test_flag in ['dev','test']:
            entire_h1c, entire_h10c, entire_h50c, entire_h100c, entire_h500c, entire_datapoints = 0, 0, 0, 0, 0, 0
            evaluated_world = DEV_WORLDS if dev_or_test_flag == 'dev' else TEST_WORLDS
            for world_name in evaluated_world: #
                reader_for_eval = WorldsReader(args=opts, world_name=world_name, token_indexers=global_toknIndexer, tokenizer=global_tokenizer)
                Evaluator = Evaluate_one_world(args=opts, world_name=world_name,
                                               reader=reader_for_eval,
                                               embedder=textfieldEmbedder,
                                               trainfinished_mention_encoder=mention_encoder,
                                               trainfinished_entity_encoder=entity_encoder,
                                               vocab=vocab, experiment_logdir=experiment_logdir,
                                               dev_or_test=dev_or_test_flag,
                                               berttokenizer=global_tokenizer,
                                               bertindexer=global_toknIndexer)
                Hits1count, Hits10count, Hits50count, Hits100count, Hits500count, data_points = Evaluator.evaluate_one_world()
                entire_h1c += Hits1count
                entire_h10c += Hits10count
                entire_h50c += Hits50count
                entire_h100c += Hits100count
                entire_h500c += Hits500count
                entire_datapoints += data_points

            dev_or_test_finallog(entire_h1c, entire_h10c, entire_h50c, entire_h100c, entire_h500c, entire_datapoints, dev_or_test_flag, experiment_logdir)

    exp_end_time = time.time()
    print('===experiment finised', exp_end_time-exp_start_time, 'sec')

if __name__ == '__main__':
    main()
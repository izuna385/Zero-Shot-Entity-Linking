import pdb, time
from utils import experiment_logger, cuda_device_parser, dev_or_test_finallog, worlds_loader
from parameters import Params
from data_reader import WorldsReader
import torch
import numpy as np
from embeddings import EmbLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from encoders import Pooler_for_mention, Pooler_for_title_and_desc
from model import Biencoder
import torch.optim as optim
from evaluator import oneLineLoaderForDevOrTestEvaluation, devEvalExperimentEntireDevWorldLog
from token_indexing import TokenIndexerReturner
from hardnegative_searcher import HardNegativesSearcherForEachEpochStart
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
    print("experiment_logdir:", experiment_logdir)
    P.dump_params(experiment_dir=experiment_logdir)
    cuda_devices = cuda_device_parser(str_ids=opts.cuda_devices)
    TRAIN_WORLDS, DEV_WORLDS, TEST_WORLDS = worlds_loader(args=opts)

    vocab = Vocabulary()
    iterator_for_training_and_evaluating_mentions = BucketIterator(batch_size=opts.batch_size_for_train,
                                                                   sorting_keys=[('context', 'num_tokens')])
    iterator_for_training_and_evaluating_mentions.index_with(vocab)

    embloader = EmbLoader(args=opts)
    emb_mapper, emb_dim, textfieldEmbedder = embloader.emb_returner()
    tokenIndexing = TokenIndexerReturner(args=opts)
    global_tokenizer = tokenIndexing.berttokenizer_returner()
    global_tokenIndexer = tokenIndexing.token_indexer_returner()

    mention_encoder = Pooler_for_mention(args=opts, word_embedder=textfieldEmbedder)
    entity_encoder = Pooler_for_title_and_desc(args=opts, word_embedder=textfieldEmbedder)
    model = Biencoder(args=opts, mention_encoder=mention_encoder, entity_encoder=entity_encoder, vocab=vocab)
    model = model.cuda()
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=opts.lr, eps=opts.epsilon,
                           weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), amsgrad=opts.amsgrad)
    devEvalEpochs = [j for j in range(1, 1000)] if opts.add_hard_negatives else \
                    [1, 3, 5] + [k * 10 for k in range(1, 100)]

    for epoch in range(opts.num_epochs):
        oneep_train_start = time.time()
        for world_name in TRAIN_WORLDS:
            reader = WorldsReader(args=opts, world_name=world_name, token_indexers=global_tokenIndexer, tokenizer=global_tokenizer)

            if opts.add_hard_negatives:
                with torch.no_grad():
                    mention_encoder.eval(), entity_encoder.eval()
                    hardNegativeSearcher = HardNegativesSearcherForEachEpochStart(args=opts, world_name=world_name,
                                                                                  reader=reader,
                                                                                  embedder=textfieldEmbedder,
                                                                                  mention_encoder=mention_encoder,
                                                                                  entity_encoder=entity_encoder, vocab=vocab,
                                                                                  berttokenizer=global_tokenizer,
                                                                                  bertindexer=global_tokenIndexer)
                    hardNegativeSearcher.hardNegativesSearcherandSetter()

            trains = reader.read('train')
            mention_encoder.train(), entity_encoder.train()
            trainer = Trainer(model=model, optimizer=optimizer,
                              iterator=iterator_for_training_and_evaluating_mentions, train_dataset=trains,
                              cuda_device=cuda_devices, num_epochs=1
                              )
            trainer.train()

        if epoch + 1 in devEvalEpochs:
            print('\n===================\n', 'TEMP DEV EVALUATION@ Epoch', epoch + 1,'\n===================\n')
            t_entire_h1c, t_entire_h10c, t_entire_h50c, t_entire_h64c, t_entire_h100c, t_entire_h500c, t_entire_datapoints \
                = oneLineLoaderForDevOrTestEvaluation(
                dev_or_test_flag='dev',
                opts=opts,
                global_tokenIndexer=global_tokenIndexer,
                global_tokenizer=global_tokenizer,
                textfieldEmbedder=textfieldEmbedder,
                mention_encoder=mention_encoder,
                entity_encoder=entity_encoder,
                vocab=vocab,
                experiment_logdir=experiment_logdir,
                finalEvalFlag=0,
                trainEpoch=epoch+1)
            devEvalExperimentEntireDevWorldLog(experiment_logdir, t_entire_h1c, t_entire_h10c,
                                               t_entire_h50c, t_entire_h64c, t_entire_h100c,
                                               t_entire_h500c, t_entire_datapoints,
                                               epoch=epoch)
        oneep_train_end = time.time()
        print('epoch {0} train time'.format(epoch+1), oneep_train_end - oneep_train_start, 'sec')

    print('====training finished=======')

    with torch.no_grad():
        model.eval()
        print('===FINAL Evaluation starts===')

        for dev_or_test_flag in ['dev','test']:
            print('\n===================\n', dev_or_test_flag, 'EVALUATION', '\n===================\n')
            entire_h1c, entire_h10c, entire_h50c, entire_h64c, entire_h100c, entire_h500c, entire_datapoints \
                = oneLineLoaderForDevOrTestEvaluation(dev_or_test_flag=dev_or_test_flag,
                                                      opts=opts,
                                                      global_tokenIndexer=global_tokenIndexer,
                                                      global_tokenizer=global_tokenizer,
                                                      textfieldEmbedder=textfieldEmbedder,
                                                      mention_encoder=mention_encoder,
                                                      entity_encoder=entity_encoder,
                                                      vocab=vocab,
                                                      experiment_logdir=experiment_logdir,
                                                      finalEvalFlag=1,
                                                      trainEpoch=-1)

            dev_or_test_finallog(entire_h1c, entire_h10c, entire_h50c, entire_h64c, entire_h100c,
                                 entire_h500c, entire_datapoints, dev_or_test_flag, experiment_logdir,
                                 )

    exp_end_time = time.time()
    print('===experiment finised', exp_end_time-exp_start_time, 'sec')
    print(experiment_logdir)


if __name__ == '__main__':
    main()

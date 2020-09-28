import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
import sys
import torch
import argparse
import os
from model_pretrain import  Plain_bert
from fairseq.models.roberta import RobertaModel
# from utils_sample import NewsIterator
# from utils_sample import cal_metric
from fairseq import utils as fairseq_utils
import utils_pretrain as utils
# import dgl
# import dgl.function as fn
#from gpu_mem_track import  MemTracker
#import inspect
#from multiprocessing import Pool
import torch.nn as nn
import math
from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
)
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from apex.parallel import DistributedDataParallel as DDP
# import apex
# from apex import amp
import torch.multiprocessing as mp
import torch.distributed as dist
random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)


#cudaid=0
metrics=['group_auc','mean_mrr','ndcg@5;10']
lr=1e-4
T_warm=5000
all_iteration=34431


def parse_args(parser):
    

    parser.add_argument("--data_dir",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--save_dir",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--data_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--test_data_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--feature_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--test_feature_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--gpu_size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--log_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--field",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--model_file",
                    type=str,
                    help="local_rank for distributed training on gpus")

#     return parser.parse_args()



# def parse_args_model(parser):
    parser.add_argument('--activation-fn',
                            choices=fairseq_utils.get_available_activation_fns(),
                            help='activation function to use')
    parser.add_argument('--dropout', type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--attention-dropout', type=float, metavar='D',
                        help='dropout probability for attention weights')
    parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                        help='dropout probability after activation in FFN.')
    parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained encoder embedding')
    parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension')
    parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension for FFN')
    parser.add_argument('--encoder-layers', type=int, metavar='N',
                        help='num encoder layers')
    parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                        help='num encoder attention heads')
    parser.add_argument('--encoder-normalize-before', action='store_true',
                        help='apply layernorm before each encoder block')
    parser.add_argument('--encoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the encoder')
    parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained decoder embedding')
    parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension')
    parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension for FFN')
    parser.add_argument('--decoder-layers', type=int, metavar='N',
                        help='num decoder layers')
    parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                        help='num decoder attention heads')
    parser.add_argument('--decoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the decoder')
    parser.add_argument('--decoder-normalize-before', action='store_true',
                        help='apply layernorm before each decoder block')
    parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                        help='decoder output dimension (extra linear layer '
                             'if different from decoder embed dim')
    parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                        help='share decoder input and output embeddings')
    parser.add_argument('--share-all-embeddings', action='store_true',
                        help='share encoder, decoder and output embeddings'
                             ' (requires shared dictionary and embed dim)')
    parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                        help='if set, disables positional embeddings (outside self attention)')
    parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                        help='comma separated list of adaptive softmax cutoff points. '
                             'Must be used with adaptive_loss criterion'),
    parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                        help='sets adaptive softmax dropout for the tail projections')
    parser.add_argument('--layernorm-embedding', action='store_true',
                        help='add layernorm to embedding')
    parser.add_argument('--no-scale-embedding', action='store_true',
                        help='if True, dont scale embeddings')
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    parser.add_argument('--no-cross-attention', default=False, action='store_true',
                        help='do not perform cross-attention')
    parser.add_argument('--cross-self-attention', default=False, action='store_true',
                        help='perform cross+self-attention')
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                        help='LayerDrop probability for encoder')
    parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                        help='LayerDrop probability for decoder')
    parser.add_argument('--encoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    parser.add_argument('--decoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                        help='iterative PQ quantization noise at training time')
    parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                        help='block size of quantization noise at training time')
    parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                        help='scalar quantization noise and scalar quantization at training time')



    return parser.parse_args()

def base_architecture(args):

    setattr(args, "encoder_embed_path", None)
    setattr(args, "encoder_embed_dim", 768)
    setattr(args, "encoder_ffn_embed_dim", 2048)
    setattr(args, "encoder_layers", 6)
    setattr(args, "encoder_attention_heads", 8)
    setattr(args, "encoder_normalize_before", False)
    setattr(args, "encoder_learned_pos", False)
    setattr(args, "decoder_embed_path", None)
    setattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    setattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    setattr(args, "decoder_layers", 6)
    setattr(args, "decoder_attention_heads", 8)
    setattr(args, "decoder_normalize_before", False)
    setattr(args, "decoder_learned_pos", False)
    setattr(args, "attention_dropout", 0.0)
    setattr(args, "activation_dropout", 0.0)
    setattr(args, "activation_fn", "relu")
    setattr(args, "dropout", 0.1)
    setattr(args, "adaptive_softmax_cutoff", None)
    setattr(args, "adaptive_softmax_dropout", 0)
    setattr(
        args, "share_decoder_input_output_embed", False
    )
    setattr(args, "share_all_embeddings", False)
    setattr(
        args, "no_token_positional_embeddings", False
    )
    setattr(args, "adaptive_input", False)
    setattr(args, "no_cross_attention", False)
    setattr(args, "cross_self_attention", False)

    setattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    setattr(args, "decoder_input_dim", args.decoder_embed_dim)

    setattr(args, "no_scale_embedding", False)
    setattr(args, "layernorm_embedding", False)
    setattr(args, "tie_adaptive_weights", False)

    print('???',args.encoder_embed_dim)

def adjust_learning_rate(optimizer,iteration,lr=lr, T_warm=T_warm, all_iteration=all_iteration ):#得看一些一共有多少个iteration再确定
    if iteration<=T_warm:
        lr=lr*float(iteration)/T_warm
    elif iteration<all_iteration:
        lr = lr * (1 - (iteration - T_warm) / (all_iteration - T_warm))
    else:
        lr=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(model,args):
    preds = []
    labels = []
    imp_indexes = []
    metrics=['group_auc']
    test_file=os.path.join(args.data_dir, args.test_data_file)  
    preds = []
    labels = []
    imp_indexes = []
    feature_file=os.path.join(args.data_dir,args.feature_file)
    iterator=NewsIterator(batch_size=1, npratio=-1,feature_file=feature_file,field=args.field,fp16=True)
    print('test...')
    cudaid=0
    #model = nn.DataParallel(model, device_ids=list(range(args.size)))
    step=0
    with torch.no_grad():
        data_batch=iterator.load_test_data_from_file(test_file,None)
        batch_t=0
        for  imp_index , user_index, his_id, candidate_id , label,_  in data_batch:
            batch_t+=len(candidate_id)
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            logit=model(his_id,candidate_id,None,mode='validation')
            # print('???',label_t,label)
            # assert 1==0
            logit=list(np.reshape(np.array(logit.cpu()), -1))
            #print('???',label)
            label=list(np.reshape(np.array(label), -1))
            imp_index=list(np.reshape(np.array(imp_index), -1))

            assert len(imp_index)==1
            imp_index=imp_index*len(logit)

            assert len(logit)==len(label)
            assert len(logit)==len(imp_index)
            assert np.sum(np.array(label))!=0


            labels.extend(label)
            preds.extend(logit)
            imp_indexes.extend(imp_index)
            step+=1
            if step%100==0:
                print('all data: ',len(labels))

    group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)
    res = cal_metric(group_labels, group_preds, metrics)
    return res['group_auc']

def train(cudaid, args,model,roberta_dict):

    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='env://',
    #     world_size=args.size,
    #     rank=cudaid)

    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)

    print('params: '," T_warm: ",T_warm," all_iteration: ",all_iteration," lr: ",lr)
    #cuda_list=range(args.size)
    print('rank: ',cudaid)
    torch.cuda.set_device(cudaid)
    model.cuda(cudaid)

    accumulation_steps=int(args.batch_size/args.size/args.gpu_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)

    # optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0,max_grad_norm=1.0)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    # model = DDP(model)
    mlm_data=utils.load_mask_data(os.path.join(args.data_dir,'data-bin/train' ),roberta_dict)



    #model = nn.DataParallel(model, device_ids=cuda_list)
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # torch.cuda.set_device(cudaid)
    
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    #model=torch.nn.parallel.DistributedDataParallel(model, device_ids=cuda_list)
    #model = torch.nn.DataParallel(model)
    #model=apex.parallel.DistributedDataParallel(model)

    accum_batch_loss=0
    #iterator=NewsIterator(batch_size=args.gpu_size, npratio=4,feature_file=os.path.join(args.data_dir,args.feature_file),field=args.field)
    #train_file=os.path.join(args.data_dir, args.data_file)  
    #for epoch in range(0,100):
    batch_t=0
    iteration=0
    print('train...',args.field)
    #w=open(os.path.join(args.data_dir,args.log_file),'w')
    if cudaid==0:
        writer = SummaryWriter(os.path.join(args.data_dir, args.log_file) )
    epoch=0
    model.train()
    # batch_t=52880-1
    # iteration=3305-1
    batch_t=0
    iteration=0
    step=0
    best_score=-1
    #w=open(os.path.join(args.data_dir,args.log_file),'w')

    # model.eval()
    # auc=test(model,args)

    for epoch in range(0,10):
    #while True:
        all_loss=0
        all_batch=0
        #data_batch=iterator.load_data_from_file(train_file,cudaid,args.size)
        data_batch=utils.get_batch(mlm_data,roberta_dict,args.batch_size,decode_dataset=None)
        for  token_list, mask_label_list, decode_label_list in data_batch:
            batch_t+=1
            #assert candidate_id.shape[1]==2
            # his_id=his_id.cuda(cudaid)
            # candidate_id= candidate_id.cuda(cudaid)
            # label = label.cuda(cudaid)
            # loss=model(his_id,candidate_id, label)

            token_list=token_list.cuda(cudaid)
            mask_label_list=mask_label_list.cuda(cudaid)
            decode_label_list=decode_label_list.cuda(cudaid)

            loss,sample_size=model(token_list,mask_label_list,decode_label_list)


            loss=loss/sample_size/math.log(2)
            print('loss: ',loss,' sample_size: ',sample_size)
            assert 1==0
            
            accum_batch_loss+=float(loss)

            all_loss+=float(loss)
            all_batch+=1

            loss = loss/accumulation_steps
            loss.backward()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            if (batch_t)%accumulation_steps==0:

                iteration+=1
                adjust_learning_rate(optimizer,iteration)
                optimizer.step()
                optimizer.zero_grad()
                if cudaid==0:
                    print(' batch_t: ',batch_t, ' iteration: ', iteration, ' epoch: ',epoch,' accum_batch_loss: ',accum_batch_loss/accumulation_steps,' lr: ', optimizer.param_groups[0]['lr'])
                    writer.add_scalar('Loss/train', accum_batch_loss/accumulation_steps, iteration)
                    writer.add_scalar('Ltr/train', optimizer.param_groups[0]['lr'], iteration)
                accum_batch_loss=0
                if iteration%5000==0 and cudaid==0:
                    torch.cuda.empty_cache()
                    model.eval()
                    if cudaid==0:
                        auc=test(model,args)
                        print(auc)
                        writer.add_scalar('auc/valid', auc, step)
                        step+=1
                        if auc>best_score:
                            torch.save(model.state_dict(), os.path.join(args.save_dir,'pretrain_best.pkl'))
                            best_score=auc
                            print('best score: ',best_score)
                    torch.cuda.empty_cache()
                    model.train()
        if cudaid==0:
            torch.save(model.state_dict(), os.path.join(args.save_dir,'Plain_robert_dot'+str(epoch)+'.pkl'))
    #w.close()
            

if __name__ == '__main__':

    # cuda_num=int(sys.argv[1])
    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)
    #main()
    parser = argparse.ArgumentParser("pretraining-model")
    args = parse_args(parser)
    #args=parse_args_model(parser)
    base_architecture(args)
    print('???',args.encoder_embed_dim)

    #roberta_dict=utils.load_dict(args.data_dir)
    roberta_dict=Dictionary.load(os.path.join(args.data_dir, 'roberta.base/dict.txt') )

    model=Plain_bert(args,roberta_dict)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
    
    # for name, param in model.named_parameters():
    #     print(name,param.shape,param.requires_grad)

    #roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file='model.pt')
    #roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file=args.model_file)

    # for name, param in roberta.named_parameters():
    #     print(name,param.shape,param.requires_grad)


    # model_dict = model.state_dict()
    # pretrained_dict={}
    # for name,parameters in roberta.named_parameters():
    #     if  'lm_head' not in name:
    #         pretrained_dict['encoder.'+name[31:]]=parameters

    # print(pretrained_dict.keys(),len(pretrained_dict.keys()))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)


    # args.world_size = args.size * 1
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '8888'
    # mp.spawn(train, nprocs=args.size, args=(args,model))

    # model.cuda(cudaid)
    train(0,args,model,roberta_dict)
    

            

























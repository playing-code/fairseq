import torch
from torch import nn
# import torch_blocksparse as tbs
# from deepscale.pt.deepscale_constants import ROUTE_TRAIN
# import types
# from base.base_model import BaseModel
# from models.bert.masked_lm_loss import MaskedLMLoss
#from models.bert.modeling_bert import BertOnlyMLMHead, BertModel
from transformers.modeling_bert import BertModel, BertConfig, BertForMaskedLM, BertOnlyMLMHead  
# from models.rankbertv2.modeling_v2 import RankBERTV2Model
# from sparse_transformers import SparseBert, SparseBertForSequenceClassification
#from models.bert.modeling_bert import BertLayerNorm
from transformers.modeling_roberta import RobertaConfig, RobertaForSequenceClassification 
import torch.nn.functional as F

import torch
from torch import nn
import torch_blocksparse as tbs

import types
#from models.bert.modeling_bert import BertOnlyMLMHead, BertModel
#from transformers.modeling_bert import BertModel, BertConfig, BertForMaskedLM, BertOnlyMLMHead  
from sparse_transformers import SparseBert, SparseRobertaForSequenceClassification



class Plain_bert(nn.Module):#
    def __init__(self,args):
        super().__init__()
        embedding_dim=768
        self.roberta=RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True,output_hidden_states=True)
        sm = SparseRobertaForSequenceClassification.from_pretrained('roberta-base')
        # sm.make_long_and_sparse(2560, "longformer", 16, True,128,list(range(0,8*20,8)))
        sm.make_long_and_sparse(2560, "longformer", 16, True,128,[0])
        self.sparse_roberta = sm.roberta
        #self.bert.config = sm.config
        #self.roberta.resize_token_embeddings(119567)
        #self.tie_weights()

        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(768)
        self.init_weights(self.dense)

        # self.atten_mask=torch.zeros((2560,2560))
        # for item in range(0,8*20,8):
        #     start=item*16
        #     #print('start: ',start)
        #     self.atten_mask[start,:]=1#global
        #     #for item2 in range(1,)
        # #global attention
        # self.atten_mask[:,0]=1

        # #print('!!!',self.atten_mask)
        # # for item in range(2560):
        # #     doc_id=int(item/128)
        # #     self.atten_mask[item][doc_id*128:doc_id+128]=1
        # # self.atten_mask=1-self.atten_mask

        # for item in range(20):
        #     #print('local: ',item*128,(item+1)*128)
        #     self.atten_mask[item*128:(item+1)*128,item*128:(item+1)*128]=1
        #     #print('!!!',item, self.atten_mask,item*128,(item+1)*128)

       # self.atten_mask=1-self.atten_mask



    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()

    def forward(self,his_id , candidate_id,label,mode='train'):

        #print('???',self.atten_mask)
        #self.atten_mask=self.atten_mask.cuda()

        batch_size,can_num,can_legth=candidate_id.shape
        batch_size,_,his_length=his_id.shape
        sample_size=candidate_id.shape[0]
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)

        his_padding_mask = 1-his_id.eq(1).type_as(his_id)#

        # print('???',his_padding_mask.shape)
        # his_padding_mask=his_padding_mask.unsqueeze(1)

        # his_padding_mask=his_padding_mask*self.atten_mask

        # print('???',his_padding_mask.dim())

        # his_atten_mask[:,self.global_mask:]=0#除local size之外的是不能看见的
        # his_atten_mask[:,self.global_mask:]=0#block里面的token除第一个之外是不能看见其他的
        # his_atten_mask[self.global_mask,:]=0#和上一行对称的mask
        # his_atten_mask[self.local_mask,:]=0#对每一个token mask掉local size中超出本篇文档的部分
        # his_atten_mask[self.local_mask,:]=0#和上一条对称

        can_padding_mask=1-candidate_id.eq(1).type_as(candidate_id)


        outputs_his = self.sparse_roberta(input_ids=his_id, attention_mask=his_padding_mask )
        #print('???',len(outputs_his.hidden_states))
        his_features=outputs_his.hidden_states[-1][:,0,:]

        outputs_can = self.roberta(input_ids=candidate_id, attention_mask=can_padding_mask, labels=None)
        can_features=outputs_can.hidden_states[-1][:,0,:]

        his_features=his_features.reshape(batch_size,1,his_features.shape[-1])
        can_features=can_features.reshape(batch_size,can_num,can_features.shape[-1]) 


        his_features = self.dense(his_features)
        his_features = self.layer_norm(his_features)

        can_features = self.dense(can_features)
        can_features = self.layer_norm(can_features)


        res=torch.matmul(his_features,can_features.transpose(1,2))
        if mode !='train':
            return res.reshape(-1)#,label.view(-1)

        res=res.reshape(-1,2)
        #print('???',res,sample_size)

        loss = F.nll_loss(
            F.log_softmax(
                res.view(-1, res.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            label.view(-1),
            reduction='sum',
            #ignore_index=self.padding_idx,
        )
        #print('loss: ',loss)
        return loss
























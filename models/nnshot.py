import sys
sys.path.append('..')
from fewshot_ner_kit.framework import FewShotNERModel
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm, trange

class NNShot(FewShotNERModel):
    
    def __init__(self,word_encoder, dot=False):
        FewShotNERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.dot = dot

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1))
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_nearest_dist__(self, embedding, tag, mask, query, q_mask):
        nearest_dist = []
        S = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == S.size(0)
        dist = self.__batch_dist__(S, query, q_mask) # [num_of_query_tokens, num_of_support_tokens]
        for label in range(torch.max(tag)+1):
            nearest_dist.append(torch.max(dist[:,tag==label], 1)[0])
        nearest_dist = torch.stack(nearest_dist, dim=1) # [num_of_query_tokens, class_num]
        return nearest_dist

    def forward(self, support, query, N, K, total_Q, batch_size=10):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        print(support['word'].shape, query['word'].shape)
        # print(support, query)
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        assert support_emb.size()[:2] == support['mask'].size()

        logits = []

        sent_query_num = query['sentence_num'][0]
        for j in trange(0,sent_query_num, batch_size):
            query_emb = self.word_encoder(query['word'][j:j+batch_size], query['mask'][j:j+batch_size]) # [num_sent, number_of_tokens, 768]
            query_emb = self.drop(query_emb)
            logits.append(self.__get_nearest_dist__(support_emb, 
                support['label'], 
                support['text_mask'],
                query_emb,
                query['text_mask'][j:j+batch_size]))

        # query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        # query_emb = self.drop(query_emb)
        # assert query_emb.size()[:2] == query['mask'].size()
        # logits.append(self.__get_nearest_dist__(support_emb, 
        #     support['label'], 
        #     support['text_mask'],
        #     query_emb,
        #     query['text_mask']))

        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        # print(logits, pred)
        return logits, pred

    
    
    

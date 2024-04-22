import numpy as np
from torch.nn import functional as F

from .utils import *
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from model.transformer_modules.TransformerEncoders import *#导入了多头注意力机制网络transformer
#from model.netvlad import NetVLAD, NetVLAD_four, NetVLAD_V2, NetVLAD_V2_Four, GAFN, GAFN_Four
#from model.netvladv2 import GAFN_V3, GAFN_Four_V3, GAFN_V3_self, GAFN_Four_V3_self, GAFN_V3_cluster, GAFN_Four_V3_cluster
#from .vivit import ViViT, ViT
#from .build import build_model, build_model3d
from transformers import BertTokenizer, BertModel
import clip




#my model begining

class InputUnitLinguistic_Bert(nn.Module):#bert初始化语言向量，再微调一个Transformer
    def __init__(self, vocab_size, layers=2,wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_Bert, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.proj_bert = nn.Sequential(
                        nn.Linear(768, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=layers)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        #self.BertEncoder = BertModel.from_pretrained("bert-base-cased")
        self.module_dim = module_dim

    def forward(self, model,questions,question_input_bert,question_mask_bert,question_ids_bert, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        # questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        # questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        # questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        with torch.no_grad():
            questions_embedding_bert = model(input_ids=question_input_bert, attention_mask=question_mask_bert, token_type_ids=question_ids_bert)
        questions_embedding = self.proj_bert(questions_embedding_bert[0].permute(1,0,2))  #将bert特征映射到模型特征长度512上
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len.squeeze())  #6层 Transformer编译文本向量     
        return questions_embedding


    
class CTVQA_Swin(nn.Module):#主模型
    def __init__(self, motion_dim, appearance_dim, module_dim, word_dim,  vocab, question_type):
        super(CTVQA_Swin, self).__init__()

        self.question_type = question_type#问题的类型
        
        self.BertEncoder = BertModel.from_pretrained("bert-base-cased")#bert编码器，编码文本特征

        encoder_vocab_size = len(vocab['question_token_to_idx'])
        
        self.num_classes = len(vocab['answer_token_to_idx'])
        self.linguistic_input_unit = InputUnitLinguistic_Bert(vocab_size=encoder_vocab_size, layers=4,wordvec_dim=word_dim,module_dim=module_dim)
        
        self.visual_input_unit = InputUnitVisual_GST_Transformer(motion_dim=motion_dim, appearance_dim=appearance_dim, layers=4,module_dim=module_dim)
        
        self.fuse_Transformer =  TransformerEncoder(embed_dim=module_dim*2, pos_flag='learned',pos_dropout=0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=4)
        
        self.space_cl_loss = NTXentLoss(module_dim,0.1)
        self.space_cl_fine_loss = NTXentLoss1(module_dim,0.1)
        
        self.mm_cl_loss = NTXentLoss_neg(module_dim,0.1)
        self.mm_cl_fine_loss = NTXentLoss1_neg(module_dim,0.1)
        
        
        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim*48, module_dim*2),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim*2),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim*2, 4))
                                       
        
        self.mm_decode=nn.Linear(module_dim*2, module_dim,bias=True)
        
    def forward(self, answer_idx, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat,\
    question_input_bert,question_mask_bert,question_ids_bert,answer_input_bert,answer_mask_bert,answer_ids_bert,ans_candidates_input_bert,ans_candidates_mask_bert,ans_candidates_ids_bert,question_bert_len,answer_bert_len,ans_candidates_bert_len,\
                    question, answer,  question_len, answer_len, appearance_dict, motion_dict,ques_type):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)，候选答案
            ans_candidates_len: [Tensor] (batch_size, 5)，候选答案的文本长度
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)#视频帧特征
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)#视频clip（一段视频帧）特征
            question: [Tensor] (batch_size, max_question_length)#问题文本
            question_len: [Tensor] (batch_size)#问题文本的长度
        return:
            logits.#返回一个答案的概率分布，一般为softMax值
        """
        
        batch_size = question.size(0)
        # get image, word, and sentence embeddings
        question_embedding_bert = self.linguistic_input_unit(self.BertEncoder,question, question_input_bert,question_mask_bert,question_ids_bert, question_bert_len)#bert编码的问题文本特征,shape:length x batch_size x channel_size

        candiates_embedding_bert=[]
        for i in range(4):
            tem_embedding_bert = self.linguistic_input_unit(self.BertEncoder,ans_candidates[:,i,:], ans_candidates_input_bert[:,i,:],\
                                       ans_candidates_mask_bert[:,i,:],ans_candidates_ids_bert[:,i,:], ans_candidates_bert_len[:,i,:])
            
            
            candiates_embedding_bert.append(tem_embedding_bert)
        
        candiates_embedding_bert=torch.stack(candiates_embedding_bert,2)#bert编码的候选答案文本特征,shape:length x batch_size x num x channel_size
        
        semantic_m,semantic_a= self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_bert)#问题视觉语义多模态特征编码
        
        loss=[]
        sp_cl_loss=self.space_cl_loss(semantic_a.permute(1,0,2),semantic_m.permute(1,0,2))
        sp_cl_fine_loss=self.space_cl_fine_loss(semantic_a.permute(1,0,2),semantic_m.permute(1,0,2))
        loss.append(sp_cl_loss)
        loss.append(sp_cl_fine_loss)       
        
        fusion_mm_fea=torch.cat((semantic_a,semantic_m),-1)
        
        fusion_mm_fea=self.fuse_Transformer(fusion_mm_fea,fusion_mm_fea)
        fusion_mm_fea=self.mm_decode(fusion_mm_fea)
        
        if self.training:
            answer_index=torch.nn.functional.one_hot(answer_idx, num_classes=4)
            ans_mask=(answer_index==1)
            true_embedding_bert=torch.masked_select(candiates_embedding_bert,ans_mask.unsqueeze(0).unsqueeze(-1))
            true_embedding_bert=true_embedding_bert.reshape(question_embedding_bert.shape[0],batch_size,\
                                                                question_embedding_bert.shape[2]).contiguous()
            
            neg_mask=(answer_index!=1)
            negative_embedding_bert=torch.masked_select(candiates_embedding_bert,neg_mask.unsqueeze(0).unsqueeze(-1))
            negative_embedding_bert=negative_embedding_bert.reshape(question_embedding_bert.shape[0],batch_size,3,\
                                                                question_embedding_bert.shape[2]).contiguous()
            padding1=torch.zeros((semantic_a.shape[0]-true_embedding_bert.shape[0],true_embedding_bert.shape[1],3,true_embedding_bert.shape[2])).cuda()
            negative_embedding_bert=torch.cat((negative_embedding_bert,padding1),0)
        
            padding=torch.zeros((semantic_a.shape[0]-true_embedding_bert.shape[0],true_embedding_bert.shape[1],true_embedding_bert.shape[2])).cuda()
            true_embedding_bert=torch.cat((true_embedding_bert,padding),0)
            
            
            
            mm_cl_loss=self.mm_cl_loss(fusion_mm_fea.permute(1,0,2),true_embedding_bert.permute(1,0,2),negative_embedding_bert.permute(1,0,2,3))
            mm_cl_fine_loss=self.mm_cl_fine_loss(fusion_mm_fea.permute(1,0,2),true_embedding_bert.permute(1,0,2),negative_embedding_bert.permute(1,0,2,3))
            #loss1=mm_cl_fine_loss+mm_cl_loss
            loss.append(mm_cl_loss)
            loss.append(mm_cl_fine_loss)
            
            score=[]
            for i in range(4):
                can_embedding_bert=candiates_embedding_bert[:,:,i,:]
                padding=torch.zeros((semantic_a.shape[0]-can_embedding_bert.shape[0],can_embedding_bert.shape[1],\
                                    can_embedding_bert.shape[2])).cuda()
                
                can_embedding_bert=torch.cat((can_embedding_bert,padding),0)
                can_embedding_bert_1=can_embedding_bert.permute(1,0,2).reshape(batch_size,-1).contiguous()
                
                fusion_mm_fea_1=fusion_mm_fea.permute(1,0,2).reshape(batch_size,-1).contiguous()
                
                tem=torch.sum(can_embedding_bert_1*fusion_mm_fea_1,-1)
                score.append(tem)
            score=torch.stack(score,-1)
            out=score#F.softmax(score,-1)
            
        else:
            score=[]
            for i in range(4):
                can_embedding_bert=candiates_embedding_bert[:,:,i,:]
                padding=torch.zeros((semantic_a.shape[0]-can_embedding_bert.shape[0],can_embedding_bert.shape[1],\
                                    can_embedding_bert.shape[2])).cuda()
                can_embedding_bert=torch.cat((can_embedding_bert,padding),0)
                can_embedding_bert_1=can_embedding_bert.permute(1,0,2).reshape(batch_size,-1).contiguous()
                fusion_mm_fea_1=fusion_mm_fea.permute(1,0,2).reshape(batch_size,-1).contiguous()
                
                tem=torch.sum(can_embedding_bert_1*fusion_mm_fea_1,-1)
                score.append(tem)
            score=torch.stack(score,-1)
            out=score#F.softmax(score,-1) 
            
        return loss,out
    
    
class InputUnitVisual_GST_Transformer(nn.Module):
    def __init__(self, motion_dim, appearance_dim, layers=2,module_dim=512):
        super(InputUnitVisual_GST_Transformer, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.QATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='learned',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=layers)
        self.QMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='learned',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=layers)
        self.VIATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='learned',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=layers)
        self.VIMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='learned',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=layers)
        self.module_dim = module_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat).permute(1,0,2)#8 x batch_Szie x module_dim
       
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat).permute(1,2,0,3)#8 x 16 x batch_Szie x module_dim
        

        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion)#32 x batch_Szie x module_dim
        
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance, 1))#32 x batch_Szie x module_dim
        

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion,question_embedding)#8 x batch_Szie x module_dim
        
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance, 1),question_embedding)#8 x batch_Szie x module_dim
        

        return torch.cat((question_visual_m,visual_embedding_m),0),\
                torch.cat((question_visual_a,visual_embedding_a),0)


class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Sequential(
                                    nn.Linear(module_dim, module_dim//2, bias=False),
                                    nn.GELU(),
                                    nn.Dropout(p=0.1),                                             
                                    )
        self.v_proj = nn.Sequential(
                                    nn.Linear(module_dim, module_dim//2, bias=False),
                                    nn.GELU(),
                                    nn.Dropout(p=0.1),                                             
                                    )

        self.cat = nn.Linear(module_dim//4, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.Sigmoid()
        self.activation1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.15)

    def forward(self, visual_feat1, visual_feat2):
        visual_feat1 = self.dropout(visual_feat1)
        visual_feat2 = self.dropout(visual_feat2)
        q_proj = self.activation1(self.q_proj(visual_feat1))
        v_proj = self.activation1(self.v_proj(visual_feat2))

        v_q_cat = torch.cat((v_proj, q_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)

        return v_q_cat    
    
#**Siam_loss

class NTXentLoss(torch.nn.Module):

    def __init__(self,  module_dim, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(module_dim, module_dim//4 , bias=True),
            nn.GELU(),
            nn.Linear( module_dim//4, module_dim//8, bias=True)
        )
        
    def forward(self, zis, zjs):
        shape=zis.shape
        zis=self.fc(zis)
        zjs=self.fc(zjs)
        zis=zis.reshape(shape[0],-1)
        zjs=zjs.reshape(shape[0],-1)
        
        
        batch_size=shape[0]
        
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(zis1,zjs1.permute(1,0))
        
        shape=similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix)
        positives = l_pos#.view(shape[0],self.batch_size, 1)
        positives = positives/self.temperature
        
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[mask].view(batch_size,batch_size-1)
        negatives =negatives /self.temperature
        
        loss=-torch.log((torch.exp(positives))/(torch.exp(positives)+torch.sum(torch.exp(negatives),-1,True)))
        return loss.sum()/(batch_size)

class NTXentLoss_neg(torch.nn.Module):

    def __init__(self,  module_dim, temperature):
        super(NTXentLoss_neg, self).__init__()
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(module_dim, module_dim//4 , bias=True),
            nn.GELU(),
            nn.Linear( module_dim//4, module_dim//8, bias=True)
        )
        
    def forward(self, zis, zjs, neg):
        shape=zis.shape
        
        zis=self.fc(zis)
        zjs=self.fc(zjs)
        zis=zis.reshape(shape[0],-1)
        zjs=zjs.reshape(shape[0],-1)
        
        n1=self.fc(neg[:,:,0,:]).reshape(shape[0],-1)
        n2=self.fc(neg[:,:,1,:]).reshape(shape[0],-1)
        n3=self.fc(neg[:,:,2,:]).reshape(shape[0],-1)
        
        
        batch_size=shape[0]
        
        
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        
        n1=F.normalize(n1, p=2, dim=-1)
        n2=F.normalize(n2, p=2, dim=-1)
        n3=F.normalize(n3, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(zis1,zjs1.permute(1,0))
        
        neg1= torch.sum(zis1*n1,-1)/self.temperature
        neg2= torch.sum(zis1*n2,-1)/self.temperature
        neg3= torch.sum(zis1*n3,-1)/self.temperature
        
        shape=similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix)
        positives = l_pos#.view(shape[0],self.batch_size, 1)
        positives = positives/self.temperature
        
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[mask].view(batch_size,batch_size-1)
        negatives =negatives /self.temperature
        
        loss=-torch.log((torch.exp(positives))/(torch.exp(positives)+torch.sum(torch.exp(negatives),-1,True)+torch.exp(neg1)+torch.exp(neg2)+torch.exp(neg3)))
        return loss.sum()/(batch_size)
    

class NTXentLoss1(torch.nn.Module):

    def __init__(self, module_dim,temperature):
        super(NTXentLoss1, self).__init__()
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(module_dim, module_dim // 4, bias=True),
            nn.GELU(),
            nn.Linear( module_dim // 4, module_dim//8, bias=True)
        )
        
    def forward(self, zis, zjs):
        shape=zis.shape
        batch_size=shape[0]
        token=shape[1]
        zis=self.fc(zis)
        zjs=self.fc(zjs)
        zis=zis.permute(1,0,2)
        zjs=zjs.permute(1,0,2)
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        similarity_matrix = torch.matmul(zis1,zjs1.permute(0,2,1))#torch.sqrt(F.relu(2-2*torch.matmul(zis1,zjs1.permute(0,2,1))))
        #similarity_matrix1 = torch.matmul(zis1,zis1.permute(0,2,1))
        shape=similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diagonal(similarity_matrix,dim1=1,dim2=2)
        positives = l_pos.view(shape[0],batch_size, 1)#torch.cat([l_pos, r_pos]).view(shape[0],2 * self.batch_size, 1)
        positives /= self.temperature
        
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[:,mask].view(shape[0],batch_size,batch_size-1)
        negatives /= self.temperature
        
        loss=-torch.log((torch.exp(positives))/(torch.exp(positives)+torch.sum(torch.exp(negatives),-1,True)))
        return loss.sum()/(batch_size*token)

class NTXentLoss1_neg(torch.nn.Module):

    def __init__(self, module_dim,temperature):
        super(NTXentLoss1_neg, self).__init__()
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(module_dim, module_dim // 4, bias=True),
            nn.GELU(),
            nn.Linear( module_dim // 4, module_dim//8, bias=True)
        )
        
    def forward(self, zis, zjs, neg):
        shape=zis.shape
        batch_size=shape[0]
        token=shape[1]
        zis=self.fc(zis)
        zjs=self.fc(zjs)
        zis=zis.permute(1,0,2)
        zjs=zjs.permute(1,0,2)
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        similarity_matrix = torch.matmul(zis1,zjs1.permute(0,2,1))#torch.sqrt(F.relu(2-2*torch.matmul(zis1,zjs1.permute(0,2,1))))
        
        
        n1=self.fc(neg[:,:,0,:]).permute(1,0,2)
        n2=self.fc(neg[:,:,1,:]).permute(1,0,2)
        n3=self.fc(neg[:,:,2,:]).permute(1,0,2)
        n1=F.normalize(n1, p=2, dim=-1)
        n2=F.normalize(n2, p=2, dim=-1)
        n3=F.normalize(n3, p=2, dim=-1)
        
        neg1= torch.sum(zis1*n1,-1,True)/self.temperature
        neg2= torch.sum(zis1*n2,-1,True)/self.temperature
        neg3= torch.sum(zis1*n3,-1,True)/self.temperature
        
        shape=similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diagonal(similarity_matrix,dim1=1,dim2=2)
        positives = l_pos.view(shape[0],batch_size, 1)#torch.cat([l_pos, r_pos]).view(shape[0],2 * self.batch_size, 1)
        positives /= self.temperature
        
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[:,mask].view(shape[0],batch_size,batch_size-1)
        negatives /= self.temperature
        
        loss=-torch.log((torch.exp(positives))/(torch.exp(positives)+torch.sum(torch.exp(negatives),-1,True)+torch.exp(neg1)+torch.exp(neg2)+torch.exp(neg3)))
        return loss.sum()/(batch_size*token)

    

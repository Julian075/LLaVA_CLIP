import math

import torch.nn as nn
import torch
import numpy as np

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(MLP, self).__init__()
        # A list of GCNConv layers
        self.linears = nn.ModuleList()
        self.linears2 = nn.ModuleList()
        self.gelu = QuickGELU()
        self.num_layers = num_layers

        if num_layers > 1:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            self.linears2.append(nn.Linear(hidden_dim, hidden_dim))

            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
                self.linears2.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=True))
            self.linears2.append(nn.Linear(output_dim, output_dim, bias=True))
        else:
            self.linears.append(nn.Linear(input_dim, output_dim))
            self.linears2.append(nn.Linear(output_dim, output_dim, bias=True))

        # A list of 1D batch normalization layers
        self.lns = nn.ModuleList()
        # self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(output_dim))
        # The log softmax layer
        self.softmax = nn.LogSoftmax()
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        for lin2 in self.linears2:
            lin2.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):  # , img_features, text_features, weight_p):
        # Pass description features through the first linear layer
        if self.num_layers > 1:
            for i, (lin, lin2) in enumerate(
                    zip(self.linears[:-1], self.linears2[:-1])):  # for i, lin in enumerate(self.linears[:-1]):
                embed1 = lin(x)
                embed2 = self.drop(lin2(self.gelu(embed1)))
                x = self.lns[i](embed1 + embed2)

                # x = nn.functional.gelu(x)
                # x = nn.functional.dropout(x, p=self.dropout, training=self.training)

            embed1 = self.linears[-1](x)
            embed2 = self.drop(self.linears2[-1](self.gelu(embed1)))
            x = self.lns[-1](embed1 + embed2)
        else:
            embed1 = self.linears[0](x)
            embed2 = self.drop(self.linears2[0](self.gelu(embed1)))
            x = self.lns[0](embed1 + embed2)

        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)

        return out


class LLaVA_CLIP(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, pretrained,pretrained_path="") -> None:
        super().__init__()
        self.description_encoder = MLP(input_dim=768, hidden_dim=hidden_dim, output_dim=512, num_layers=num_layers,
                                       dropout=dropout, return_embeds=True)
        self.criterion=nn.CrossEntropyLoss(reduction="mean")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # temperature
        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_LLaVA = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if pretrained:
            self.pesos_preentrenados = torch.load(pretrained_path)#('weights/best_model_params_path.pth')
            self.description_encoder.load_state_dict(self.pesos_preentrenados)

    # LLaVA-CLIP Loss
    def LLaVA_CLIP_loss(self, logits: torch.Tensor,label,t) :
        loss_i=0
        for b in range(len(logits)):
            num = torch.exp(logits[b][label[b]]/t)
            dem = torch.sum(torch.exp(logits[b]/t))
            loss_i+=torch.log(num/dem)
        loss=-loss_i/len(logits)
        return loss

    def LLaVA_CLIP_loss2(self, logits: torch.Tensor, labels,t):
        temperature=t
        # creo un dict para saber en el batch cuales inputs tienen los mismos target labels de salida
        inputs_expected_class = {}
        for index in range(len(labels)):
            clas = labels[index]
            if not (clas in inputs_expected_class.keys()):
                inputs_expected_class[clas] = [index]
            else:
                inputs_expected_class[clas].append(index)

        # Iterar sobre tod.o el batch
        loss = 0.00
        for category in inputs_expected_class.keys():
            # iterar sobre inputs con mismo label
            aux_loss = 0.00
            for inputs_index in inputs_expected_class[category]:
                num = torch.exp((logits[inputs_index][labels[inputs_index]])/temperature)
                dem = torch.sum(torch.exp(logits[inputs_index]/temperature))
                aux_loss += torch.log(num / dem)
            loss += -aux_loss / len(inputs_expected_class[category])

        return loss

        # LLaVA-CLIP  Accuracy

    def LLaVA_CLIP_acc(self, img_feat,description_feat,text_feat,weight_p,target_ind):
        sim_clip= img_feat @ text_feat
        sim_clip = sim_clip / sim_clip.norm(dim=-1, keepdim=True)

        sim_bert = description_feat.half()  @ text_feat
        sim_bert = sim_bert / sim_bert.norm(dim=-1, keepdim=True)

        sim_total=sim_clip*weight_p+sim_bert*(1-weight_p)
        sim_total= sim_total / sim_total.norm(dim=-1, keepdim=True)

        predicted_index = torch.argmax(sim_total, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc

    def forward(self, embeddings, img_features, txt_features, weight_p,target_ind,temp):
        #text_features=txt_features[0]
        #batch_text=txt_features[1]
        description_features = self.description_encoder(embeddings)
        description_features = description_features / description_features.norm(dim=-1, keepdim=True)

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (img_features @ txt_features) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        logit_scale_LLaVA = self.logit_scale_LLaVA.exp()
        similarity_bert = (description_features.half() @ txt_features) * logit_scale_LLaVA
        similarity_bert = similarity_bert / similarity_bert.norm(dim=-1, keepdim=True)

        similarity = (similarity_clip * weight_p + similarity_bert * (1 - weight_p))
        out_logits = similarity / similarity.norm(dim=-1, keepdim=True)

        # with torch.no_grad():
        #    output_indices = output_indices.squeeze()

        loss = self.LLaVA_CLIP_loss(out_logits,target_ind,temp)
        #loss = self.LLaVA_CLIP_loss2(out_logits, target_ind,temp)
        acc = self.LLaVA_CLIP_acc(img_features,description_features,txt_features,weight_p,target_ind)
        return loss, acc
from torch import nn
import torch 
import einops
import numpy as np
from transformers import BertConfig, BertModel
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from ..util.logger import get_tqdm

class CausalBertModel(BertModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, 
        input_ids: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if causal_attention_mask is None:
            raise ValueError("causal_attention_mask is required")
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        #elif input_ids is not None:
        #    self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        #    input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # Original: extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        extended_attention_mask = causal_attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        
 
class BaseBert(nn.Module):
    def __init__(self,
        smst_names_list: list,
        n_latent: int = 32,
        n_layers: int = 3,
        n_heads: int = 8,
        n_hidden: int = 1024,
        dropout: float = 0.1,
        device: str = "cpu",
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        vocab_size: int = 5981,
        hidden_dropout_prob: float = 0.1,
        num_attention_heads: int = 4,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 5000,
        num_hidden_layers: int = 4,
        intermediate_size: int = 512,
        attention_probs_dropout_prob: float = 0.1,
        
    ):
        super().__init__()
        
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.smst_names_list = smst_names_list
        
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_attention_heads = num_attention_heads
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        bert_config = BertConfig.from_dict(
            {
                "hidden_size": 2*self.n_latent,
                "hidden_act": self.hidden_act,
                "initializer_range": self.initializer_range,
                "vocab_size": self.vocab_size,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "num_attention_heads": self.num_attention_heads,
                "type_vocab_size": self.type_vocab_size,
                "max_position_embeddings": self.max_position_embeddings,
                "num_hidden_layers": self.num_hidden_layers,
                "intermediate_size": self.intermediate_size,
                "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            }
        )

        self.model = CausalBertModel(bert_config)
        self.linear_layer = nn.Linear(
            in_features=2*self.n_latent, 
            out_features=1
        ) 
        self.device = device
        self.to(device)
        
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def bert_preprocess(self,
                        expression_matrix : torch.Tensor = None):
        self.smst_embedding = nn.Embedding(len(self.smst_names_list),
                                           self.n_latent)
        smst_max = 10.
        smst_min = 0.
        smst_bins = torch.linspace(smst_min, 
                                    smst_max, 
                                    self.n_latent).to(self.device)
        
        smst_upper = torch.cat([smst_bins[1:], 
                                smst_bins.new_tensor([1e6])
                                ], dim=-1).to(self.device)
        smst_embedding = self.smst_embedding(
                    torch.arange(len(self.smst_names_list))
                ).to(self.device)
        expression_grame = (expression_matrix.unsqueeze(-1) > smst_bins) * (expression_matrix.unsqueeze(-1) < smst_upper)
        bert_input = torch.concat(
            [
                expression_grame,
                einops.repeat(smst_embedding, "g d -> b g d", b=expression_matrix.shape[0]),
                ],
            dim=-1,
            )
        return bert_input      
        
    def mask_attention(self, 
                       mask_ratio: float = 0.15,
                       expression_matrix : torch.Tensor = None):
        self.mask_ratio = mask_ratio
        batch_size, n_genes = expression_matrix.shape
        attention_mask = (torch.rand((batch_size,n_genes)) > self.mask_ratio).to(
            self.device
        ).to(torch.float32)
        masked_matrix = expression_matrix * attention_mask
        return masked_matrix, attention_mask
    
    def forward(self,
        x: torch.Tensor,
        causal_attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        bert_output = self.model(inputs_embeds=x,
                                 causal_attention_mask=causal_attention_mask,
                                 output_attentions=output_attentions)
        matrix_output = self.linear_layer(bert_output.last_hidden_state)   
        return {
            "matrix_output": matrix_output,
            "bert_output": bert_output,
        }
        
    def fit(self, 
        expression_matrix : torch.Tensor,
        mask_ratio: float = 0.15,
        epochs: int = 30,
        n_per_epoch: int = 64,
        lr: float = 1e-3,
    ):
        self.train()
        #epoch_loss = []
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        pbar = get_tqdm()(range(epochs), desc="Epoch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for epoch in range(0,epochs):
            total_batch_size = expression_matrix.shape[0]
            pbar.desc = "Epoch {}".format(epoch)
            for i in range(0, total_batch_size, n_per_epoch):
                expression_matrix_perbatch = expression_matrix[i:i+n_per_epoch]
                expression_matrix_perbatch = expression_matrix_perbatch.to(self.device)
                masked_matrix, masked_token_mask = self.mask_attention(
                    mask_ratio, 
                    expression_matrix_perbatch
                )
                causal_attention_mask = torch.ones((
                    masked_matrix.shape[0], masked_matrix.shape[1], masked_matrix.shape[1]
                )).to(torch.bool).to(self.device)
                
                causal_attention_mask[:,:int(masked_matrix.shape[1]/2),:int(masked_matrix.shape[1]/2)] = False
                causal_attention_mask[:,int(masked_matrix.shape[1]/2):,int(masked_matrix.shape[1]/2):] = False
                
                causal_attention_mask = causal_attention_mask * masked_token_mask.unsqueeze(1)
                causal_attention_mask = (
                    causal_attention_mask.permute(0, 2, 1) * masked_token_mask.unsqueeze(1)
                ).permute(0, 2, 1)
                
                optimizer.zero_grad()
                bert_input = self.bert_preprocess(masked_matrix)
                
                output = self.forward(bert_input,causal_attention_mask=causal_attention_mask)
                
                criteria = nn.MSELoss()
                loss = criteria(output["matrix_output"].squeeze(), expression_matrix_perbatch)
                loss.backward()
                optimizer.step()
                #epoch_loss.append(loss.item())
            pbar.set_postfix({"loss": loss.item()
            }) 
            pbar.update(1)           
        pbar.close()        
        #return epoch_loss
    @torch.no_grad()            
    def get_bert_expression_martix(self,
                                   expression_matrix : torch.Tensor,
                                   mask_ratio: float = 0.15,
                                   n_per_epoch: int = 64):
        total_batch_size = expression_matrix.shape[0]
        expression_matrix_output =  []
        all_attention_mask = []
        for i in range(0, total_batch_size, n_per_epoch):
            expression_matrix_perbatch = expression_matrix[i:i+n_per_epoch]
            expression_matrix_perbatch = expression_matrix_perbatch.to(self.device)
            
            masked_matrix, masked_token_mask = self.mask_attention(
                mask_ratio, 
                expression_matrix_perbatch
            )
            causal_attention_mask = torch.ones((
                masked_matrix.shape[0], masked_matrix.shape[1], masked_matrix.shape[1]
            )).to(torch.bool).to(self.device)
            
            causal_attention_mask[:,:int(masked_matrix.shape[1]/2),:int(masked_matrix.shape[1]/2)] = False
            causal_attention_mask[:,int(masked_matrix.shape[1]/2):,int(masked_matrix.shape[1]/2):] = False
            
            causal_attention_mask = causal_attention_mask * masked_token_mask.unsqueeze(1)
            causal_attention_mask = (
                causal_attention_mask.permute(0, 2, 1) * masked_token_mask.unsqueeze(1)
            ).permute(0, 2, 1)
                        
            bert_input = self.bert_preprocess(masked_matrix)
            output = output = self.forward(bert_input,causal_attention_mask=causal_attention_mask)
            
            expression_matrix_output.append(
                output["matrix_output"].squeeze().detach().cpu().numpy()
            )
            all_attention_mask.append(
                masked_token_mask.detach().cpu().numpy()
            )
        expression_matrix_output = np.concatenate(expression_matrix_output, axis=0)
        all_attention_mask = np.concatenate(all_attention_mask, axis=0)
        return expression_matrix_output, all_attention_mask
     
    @torch.no_grad()    
    def get_attentions(self,
                        expression_matrix : torch.Tensor,
                        n_per_epoch: int = 64):
        total_batch_size = expression_matrix.shape[0]
        all_attentions = []
        for i in range(0, total_batch_size, n_per_epoch):
            expression_matrix_perbatch = expression_matrix[i:i+n_per_epoch]
            expression_matrix_perbatch = expression_matrix_perbatch.to(self.device)
            causal_attention_mask = torch.ones((
                expression_matrix_perbatch.shape[0], expression_matrix_perbatch.shape[1], expression_matrix_perbatch.shape[1]
            )).to(torch.bool).to(self.device)
            masked_token_mask = torch.ones((
                expression_matrix_perbatch.shape[0], expression_matrix_perbatch.shape[1]
            )).to(torch.bool).to(self.device)
            causal_attention_mask[:,:int(expression_matrix_perbatch.shape[1]/2),:int(expression_matrix_perbatch.shape[1]/2)] = False
            causal_attention_mask[:,int(expression_matrix_perbatch.shape[1]/2):,int(expression_matrix_perbatch.shape[1]/2):] = False
            causal_attention_mask = causal_attention_mask * masked_token_mask.unsqueeze(1)
            causal_attention_mask = (
                causal_attention_mask.permute(0, 2, 1) * masked_token_mask.unsqueeze(1)
            ).permute(0, 2, 1) 
            #return causal_attention_mask                     
            bert_input = self.bert_preprocess(expression_matrix_perbatch)
            output = self.forward(bert_input,
                                  causal_attention_mask=causal_attention_mask, 
                                  output_attentions=True)
            all_attentions.append(
                output["bert_output"].attentions[0].detach().cpu().numpy()
            )
        all_attentions = np.concatenate(all_attentions, axis=0)
        return all_attentions        
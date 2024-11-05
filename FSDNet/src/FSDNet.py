import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation


class FSDNet(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FSDNet",
                 gpu=-1,
                 model_structure="parallel",
                 learning_rate=1e-3, 
                 embedding_dim=16,
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 choice=0,
                 loss_coefficient=0.3,
                 fea_coefficient=0.01,
                 noise_ratio=0,
                 Temp=1,
                 **kwargs):
        super(FSDNet, self).__init__(feature_map,
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        self.model_structure = model_structure
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None,
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1)
        self.output_layers = nn.ModuleList([
            nn.Linear(final_dim, 1) for unit in parallel_dnn_hidden_units
        ])
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.loss_coefficient = loss_coefficient
        self.choice = choice
        self.fea_coefficient = fea_coefficient
        self.T = Temp
        self.noise_ratio = noise_ratio

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, dynamic_emb_dim=True)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out_all = self.crossnet(flat_feature_emb)
        if self.model_structure == "parallel":
            dnn_out_all = self.parallel_dnn(flat_feature_emb)
            concatenated_outputs = []
            for cross_o, dnn_o in zip(cross_out_all, dnn_out_all):
                concatenated_output = torch.cat((cross_o, dnn_o), dim=-1)
                concatenated_outputs.append(concatenated_output)
        y_pred_list_o=[]
        for fc, con_out in zip(self.output_layers, concatenated_outputs):
            y_p = fc(con_out)
            y_pred_list_o.append(y_p)
        y_pred_list = []
        for y_po in y_pred_list_o:
            y_po = self.output_activation(y_po)
            y_pred_list.append(y_po)

        y_pred = sum(y_pred_list)/len(y_pred_list)
        return_dict = {"y_pred": y_pred, 'concatenated_outputs': concatenated_outputs, 'y_pred_list': y_pred_list, 'y_pred_o': y_pred_list_o}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred_list_o = return_dict['y_pred_o']
        concatenated_outputs=return_dict['concatenated_outputs']
        y_pred_list= return_dict['y_pred_list']
        ce_loss = [self.loss_fn(pred, y_true, reduction='mean') for pred in y_pred_list]
        ce_total_loss = sum(ce_loss)

        teacher_outputs_o = y_pred_list_o[-1]
        distillation_soft = nn.KLDivLoss(reduction='batchmean')
        soft_losses_o_1 = [distillation_soft(self.output_activation(pred/self.T).log(), self.output_activation(teacher_outputs_o/self.T))
                               for pred in y_pred_list_o[:-1]]
        soft_losses_o_2 = [distillation_soft(self.output_activation(1.0 - pred / self.T).log(),
                                           self.output_activation((1.0 - teacher_outputs_o) / self.T))
                         for pred in y_pred_list_o[:-1]]
        soft_losses_o= soft_losses_o_1 + soft_losses_o_2
        soft_total_loss = sum(soft_losses_o)

        feature_losses = []
        teacher_feature = concatenated_outputs[-1]
        loss_c = nn.MSELoss()
        for output in concatenated_outputs[:-1]:
            distances = loss_c(teacher_feature, output)
            feature_losses.append(distances)
        feature_total_loss = sum(feature_losses)

        if self.choice == 0:
            total_loss = ce_total_loss * self.loss_coefficient + soft_total_loss*(1-self.loss_coefficient)
        elif self.choice == 1:
            total_loss = ce_total_loss + self.fea_coefficient * feature_total_loss
        elif self.choice == 2:
            total_loss = ce_total_loss * self.loss_coefficient + soft_total_loss*(1-self.loss_coefficient) +  self.fea_coefficient * feature_total_loss
        else:
            raise ValueError("Invalid choice. Please select 0, 1, or 2.")

        return total_loss


class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0
        outputs = []
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
            outputs.append(X_i)
        return outputs


class MLP_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None,
                 dropout_rates=0.0,
                 batch_norm=False,
                 layer_norm=False,
                 norm_before_activation=True,
                 use_bias=True):
        super(MLP_Block, self).__init__()
        self.layers = nn.ModuleList()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            layer_components = []
            linear_layer = nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias)
            layer_components.append(linear_layer)
            if norm_before_activation:
                if batch_norm:
                    layer_components.append(nn.BatchNorm1d(hidden_units[idx + 1]))
                elif layer_norm:
                    layer_components.append(nn.LayerNorm(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                layer_components.append(hidden_activations[idx])
            if not norm_before_activation:
                if batch_norm:
                    layer_components.append(nn.BatchNorm1d(hidden_units[idx + 1]))
                elif layer_norm:
                    layer_components.append(nn.LayerNorm(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                layer_components.append(nn.Dropout(p=dropout_rates[idx]))
            self.layers.append(nn.Sequential(*layer_components))

        output_layer=[]
        if output_dim is not None:
            output_layer = [nn.Linear(hidden_units[-1], output_dim, bias=use_bias)]
        if output_activation:
            output_layer.append(get_activation(output_activation))
            self.layers.append(nn.Sequential(*output_layer))

    def forward(self, inputs):
        x = inputs
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


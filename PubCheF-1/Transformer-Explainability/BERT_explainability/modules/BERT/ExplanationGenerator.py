import argparse
import gc
import numpy as np
import torch
import glob
import time


# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention



class BatchGenerator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    # # NOTE Modified? from 7/15/24
    # def generate_LRP(self, input_ids, attention_mask, indices=None, start_layer=11):
    #     kwargs = {"alpha": 1}

    #     rollout_all = None
    #     for i_, index in enumerate(indices):
    #         output = self.model(input_ids=input_ids[i_], attention_mask=attention_mask[i_])[0]
    #         print(output)
    #         raise "stop"
    #         output_i = output[0, index]
    #         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    #         one_hot[0, index] = 1
    #         one_hot_vector = one_hot
    #         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    #         one_hot = torch.sum(one_hot.cuda() * output_i)

    #         one_hot.backward(retain_graph=True)
    #         self.model.zero_grad()
    #         self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

    #         cams = []
    #         blocks = self.model.bert.encoder.layer
    #         for blk in blocks:
    #             grad = blk.attention.self.get_attn_gradients()
    #             cam = blk.attention.self.get_attn_cam()
    #             cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
    #             grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
    #             cam = grad * cam
    #             cam = cam.clamp(min=0).mean(dim=0)
    #             cams.append(cam.unsqueeze(0))

    #         rollout = compute_rollout_attention(cams, start_layer=start_layer)
    #         rollout[:, 0, 0] = rollout[:, 0].min()

    #         if rollout_all is None:
    #             rollout_all = rollout[:, 0].unsqueeze(0)  # Initialize rollout_all with the correct shape and device
    #         else:
    #             rollout_all = torch.concat([rollout_all, rollout[:, 0].unsqueeze(0)], dim=0)

    #     return rollout_all

    # NOTE Batch LRP # version as of 7/15/24 9:30am
    def generate_LRP(self, input_ids, attention_mask, indices=None, start_layer=11):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}

        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        for i_, index in enumerate(indices):
            one_hot[i_, index] = 1

        one_hot_vector = one_hot
        one_hot = torch.tensor(one_hot, device=output.device, requires_grad=True)
        one_hot = (one_hot * output).sum(dim=1)

        rollout_all = None
        for i, scalar in enumerate(one_hot):
            scalar.backward(retain_graph=(i < len(one_hot) - 1))
            self.model.zero_grad()
            self.model.relprop(torch.tensor(one_hot_vector[i]).to(input_ids.device), **kwargs)

            cams = []
            for blk in self.model.bert.encoder.layer:
                grad = blk.attention.self.get_attn_gradients().detach()
                cam = blk.attention.self.get_attn_cam().detach()
                cam = cam[i].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[i].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))

            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            rollout[:, 0, 0] = rollout[:, 0].min()

            if rollout_all is None:
                rollout_all = rollout[:, 0].unsqueeze(0)  # Initialize rollout_all with the correct shape and device
            else:
                rollout_all = torch.concat([rollout_all, rollout[:, 0].unsqueeze(0)], dim=0)

        return rollout_all


    # def generate_LRP(self, input_ids, attention_mask, indices=None, start_layer=11):
    #     output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
    #     kwargs = {"alpha": 1}

    #     # # Setup one-hot encoding for all indices at once
    #     # one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32) # batch size x # labels
    #     # for i_, index in enumerate(indices):
    #     #     one_hot[i_, index] = 1 # fill in one hot at the correct index for each sample
    #     # one_hot_tensor = torch.tensor(one_hot, device=output.device, requires_grad=True) # convert to tensor

    #     # # Backprop all classes at once
    #     # one_hot_output = (one_hot_tensor * output).sum(dim=1) # sum over all labels to just get vector of dim batch size
    #     # one_hot_output.backward(torch.ones_like(one_hot_output)) # backpropagate the specified class prob
    #     # self.model.zero_grad()

    #     # # Single call to relprop for all classes
    #     # self.model.relprop(one_hot_tensor, **kwargs) # relprop for all classes

    #     one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
    #     for (i_, index) in enumerate(indices):
    #         one_hot[i_, index] = 1
    #     one_hot_vector = one_hot
    #     one_hot = torch.tensor(one_hot, device=output.device, requires_grad=True)
    #     one_hot = (one_hot * output).sum(dim=1)


    #     # Collect and process the CAMs
    #     rollout_all = []
    #     # for i, _ in enumerate(one_hot_tensor):
    #     for i, scalar in enumerate(one_hot):
    #         # clear gradients on tensor?
    #         self.model.zero_grad()

    #         scalar.backward(retain_graph=True)

    #         self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

    #         cams = []
    #         for blk in self.model.bert.encoder.layer:
    #             grad = blk.attention.self.get_attn_gradients().detach()
    #             cam = blk.attention.self.get_attn_cam().detach()
    #             cam = cam[i].reshape(-1, cam.shape[-1], cam.shape[-1])
    #             grad = grad[i].reshape(-1, grad.shape[-1], grad.shape[-1])
    #             cam = grad * cam
    #             cam = cam.clamp(min=0).mean(dim=0)
    #             cams.append(cam.unsqueeze(0))

    #         rollout = compute_rollout_attention(cams, start_layer=start_layer)
    #         rollout[:, 0, 0] = rollout[:, 0].min()
    #         rollout_all.append(rollout[:, 0])

    #         # clear graph
    #         scalar.grad = None

    #     # Concatenate all rollouts
    #     rollout_all = torch.stack(rollout_all)

    #     return rollout_all




    # def generate_LRP(self, input_ids, attention_mask, indices=None, start_layer=11):
    #     output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
    #     kwargs = {"alpha": 1}
    #     # if index == None:
    #     #     raise
    #         # index = np.argmax(output.cpu().data.numpy(), axis=-1)

    #     one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
    #     for (i_, index) in enumerate(indices):
    #         one_hot[i_, index] = 1
    #     one_hot_vector = one_hot
    #     one_hot = torch.tensor(one_hot, device=output.device, requires_grad=True)
    #     one_hot = (one_hot * output).sum(dim=1)

    #     rollout_all = []
    #     for i, scalar in enumerate(one_hot):
    #         scalar.backward(retain_graph=(i < len(one_hot) - 1))
    #         self.model.zero_grad()

    #         self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

    #         cams = []
    #         blocks = self.model.bert.encoder.layer
    #         for blk in blocks:
    #             grad = blk.attention.self.get_attn_gradients().detach()
    #             cam = blk.attention.self.get_attn_cam().detach()
    #             cam = cam[i].reshape(-1, cam.shape[-1], cam.shape[-1])
    #             grad = grad[i].reshape(-1, grad.shape[-1], grad.shape[-1])
    #             cam = grad * cam
    #             cam = cam.clamp(min=0).mean(dim=0)
    #             cams.append(cam.unsqueeze(0))
    #         rollout = compute_rollout_attention(cams, start_layer=start_layer)
    #         rollout[:, 0, 0] = rollout[:, 0].min()
    #         rollout = rollout.detach().cpu()
    #         if rollout_all is None:
    #             rollout_all = rollout[:, 0]
    #         else:
    #             rollout_all = torch.concat([rollout_all, rollout[:, 0]], dim=0)
    #         scalar.grad = None
    #     output = output.detach().cpu()
    #     one_hot = one_hot.detach().cpu()
    #     input_ids = input_ids.detach().cpu()
    #     attention_mask = attention_mask.detach().cpu()
    #     cam = cam.detach().cpu()
    #     grad = grad.detach().cpu()
    #     rollout = rollout.detach().cpu()
    #     cams = [cam.detach().cpu() for cam in cams]
    #     # blocks = blocks.detach().cpu()
    #     scalar = scalar.detach().cpu()
    #     del output, one_hot, input_ids, attention_mask, cam, grad, rollout, cams, blocks, scalar, one_hot_vector, kwargs, index
    #     self.model.to("cpu")

    #     torch.cuda.empty_cache()

    #     return rollout_all


    # def wipe_memory(self): # DOES WORK
    #     self._model_to(torch.device('cpu'))
    #     del self.model
    #     gc.collect()
    #     torch.cuda.empty_cache()

    # def _model_to(self, device):
    #     for param in self.model.parameters():
    #         # Not sure there are any global tensors in the state dict
    #         if isinstance(param, torch.Tensor):
    #             param.data = param.data.to(device)
    #             if param._grad is not None:
    #                 param._grad.data = param._grad.data.to(device)
    #         elif isinstance(param, dict):
    #             for subparam in param.values():
    #                 if isinstance(subparam, torch.Tensor):
    #                     subparam.data = subparam.data.to(device)
    #                     if subparam._grad is not None:
    #                         subparam._grad.data = subparam._grad.data.to(device)


    # def generate_LRP(self, input_ids, attention_mask, index=None, start_layer=11):
    #     # use grad but only here
    #     with torch.enable_grad():
    #         output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

    #     output = output.cpu()
    #     # move to cpu
    #     self.model.zero_grad()
    #     self.model.to("cpu")
        
    #     # for each attribute in self.model, delete and remove from gpu memory
    #     for name, param in self.model.named_parameters():
    #         try:
    #             # move to cpu
    #             param = param.cpu()
    #         except:
    #             pass
        
    #     # raise "stop"
    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     # raise "stop"

    #     # kwargs = {"alpha": 1}
    #     # if index is None:
    #     #     index = output.argmax(dim=-1).cpu().numpy()

    #     # one_hot = np.zeros((output.shape[0], output.size(-1)), dtype=np.float32)
    #     # one_hot[np.arange(output.shape[0]), index] = 1
    #     # one_hot = torch.tensor(one_hot, device=output.device, requires_grad=True)
    #     # one_hot_output = (one_hot * output).sum(dim=1)

    #     rollout_all = []
    #     # for i, scalar in enumerate(one_hot_output):
    #         # scalar.backward(retain_graph=(i < len(one_hot_output) - 1))
    #         # self.model.zero_grad()
    #         # self.model.relprop(one_hot.clone().detach(), **kwargs)

    #         # cams = []
    #         # for blk in self.model.bert.encoder.layer:
    #         #     grad = blk.attention.self.get_attn_gradients().detach()
    #         #     cam = blk.attention.self.get_attn_cam().detach()
    #         #     cam = (grad[i] * cam[i]).clamp(min=0).mean(dim=0)
    #         #     cams.append(cam.unsqueeze(0))

    #         # rollout = compute_rollout_attention(cams, start_layer=start_layer)
    #         # rollout[:, 0, 0] = rollout[:, 0].min()
    #         # rollout_all.append(rollout[0].cpu())  # Only append the needed tensor, reduced to CPU

    #         # # clear graph
    #         # scalar.grad = None
            
    #     # return torch.stack(rollout_all)

    #     # output.grad = None
    #     return torch.tensor([0] * output.shape[0], requires_grad=False)
    




class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    # NOTE Original (single input)
    def generate_LRP(self, input_ids, attention_mask, index=None, start_layer=11):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cams = []
        blocks = self.model.bert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        rollout[:, 0, 0] = rollout[:, 0].min()
        return rollout[:, 0]

    def generate_LRP_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0]
        cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0]

    def generate_full_lrp(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        cam = self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)
        cam = cam.sum(dim=2)
        cam[:, 0] = 0
        return cam

    def generate_attn_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        cam = cam.mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0]

    def generate_rollout(self, input_ids, attention_mask, start_layer=0, index=None):
        self.model.zero_grad()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        blocks = self.model.bert.encoder.layer
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attention.self.get_attn()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        rollout[:, 0, 0] = 0
        return rollout[:, 0]

    def generate_attn_gradcam(self, input_ids, attention_mask, index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()
        grad = self.model.bert.encoder.layer[-1].attention.self.get_attn_gradients()

        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam[:, 0, 0] = 0
        return cam[:, 0]


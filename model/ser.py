# Copyright 2020-present, Tao Zhuo
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import torchvision.transforms.functional as ttf

from utils.buffer import Buffer

def tf_tensor(xs, transforms):
    device = xs.device

    xs = torch.cat([transforms(x).unsqueeze_(0) 
                    for x in xs.cpu()], dim=0)

    return xs.to(device=device)

class SER:
    def __init__(self, net, args):
        super(SER, self).__init__()                
        self.net = net
        self.net_old = None
        self.optim = None        
        
        self.args = args
        self.buffer = Buffer(args.buffer_size, args.device)

    def end_task(self):          
        self.net_old = deepcopy(self.net)
        self.net_old.eval()        
        
    def observe(self, inputs, labels):

        self.optim.zero_grad()
                        
        inputs_aug = tf_tensor(inputs, self.args.transform)    
        outputs = self.net(inputs_aug)
        loss = F.cross_entropy(outputs, labels)
                
        if self.net_old is not None:
            if self.args.setting == 'domain_il': 
                augment = None
            else:
                augment = self.args.transform
            
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(inputs.size(0), transform=augment)
            buf_outputs = self.net(buf_inputs)
            loss += F.cross_entropy(buf_outputs, buf_labels)            
                     
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            
            outputs_old = self.net_old(inputs_aug)
            loss += self.args.beta * F.mse_loss(outputs, outputs_old)                           

        loss.backward()
        self.optim.step()
                
        if self.args.setting == 'domain_il': 
            self.buffer.add_data(examples=inputs_aug, labels=labels, logits=outputs.data)
        else:
            self.buffer.add_data(examples=inputs, labels=labels, logits=outputs.data)
        
        return loss
 
    
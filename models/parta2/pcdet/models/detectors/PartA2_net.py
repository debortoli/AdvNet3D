from .detector3d_template import Detector3DTemplate

import pdb
import numpy as np
import torch

class PartA2Net(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, data_type, use_adversarial_discriminator):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if data_type == 'airsim':
            model_cfg['POINT_HEAD']['TARGET_CONFIG']['BOX_CODER_CONFIG']['mean_size'] =\
                 model_cfg['POINT_HEAD']['TARGET_CONFIG']['BOX_CODER_CONFIG']['mean_size_airsim']
        else:
            model_cfg['POINT_HEAD']['TARGET_CONFIG']['BOX_CODER_CONFIG']['mean_size'] =\
                 model_cfg['POINT_HEAD']['TARGET_CONFIG']['BOX_CODER_CONFIG']['mean_size_real']

        self.use_adversarial_discriminator = use_adversarial_discriminator
        self.module_list = self.build_networks(use_adversarial_discriminator)

    def forward(self, batch_dict):
        self.init_batch_dict = batch_dict

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, loss_ew, tb_dict, disp_dict = self.get_training_loss(batch_dict['adv_lambda_weighting'])

            return loss, loss_ew, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, adv_lambda_weighting, loss_args=None):
        disp_dict = {}
        if self.dense_head is not None:
            # its present for the anchor network but not the -free version
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss_rpn + loss_point + loss_rcnn

        else:
            loss_point, loss_point_ew, tb_dict = self.point_head.get_loss(self.init_batch_dict['num_voxels_per_example'])
            loss_rcnn, loss_rcnn_ew, tb_dict = self.roi_head.get_loss(self.init_batch_dict['num_voxels_per_example'], tb_dict)
            if self.use_adversarial_discriminator:
                #loss_adv, loss_adv_ew = self.get_adv_loss(tb_dict, adv_lambda_weighting)
                loss_adv, loss_adv_ew = self.get_emd_loss(tb_dict, adv_lambda_weighting)
            else:
                loss_adv, loss_adv_ew = 0, 0
            #print(loss_adv)
            loss =  loss_point + loss_rcnn + loss_adv
            loss_ew = loss_point_ew + loss_rcnn_ew  #+ loss_adv_ew
            
            diff = abs(loss.item() - loss_ew.sum())
            if (np.round(loss.item(),3)*0.05 > 0) and (np.round(loss.item(),3)*0.05 < diff):
                print('Batchwise computation seems to have broken')
                #pdb.set_trace()

        tb_dict['total_loss'] = loss

        return loss, loss_ew, tb_dict, disp_dict

    def get_adv_loss(self, tb_dict, adv_lambda_weighting):
        '''
        '''
        m = torch.nn.Sigmoid()
        adv_loss = torch.nn.BCELoss(reduction='none')
        loss_ew = adv_loss(m(self.init_batch_dict['adv_classification_output']), self.init_batch_dict['data_type'].float())
        loss = loss_ew.sum()        

        loss *= adv_lambda_weighting
        loss_ew *= adv_lambda_weighting
        tb_dict['adv_lambda_weighting'] = adv_lambda_weighting
        tb_dict['loss_adv'] = loss.item()

        return loss, loss_ew
    def get_emd_loss(self, tb_dict, adv_lambda_weighting):
        syn_inds = (self.init_batch_dict['data_type']==0).nonzero()
        real_inds = (self.init_batch_dict['data_type']==1).nonzero()

        m = torch.nn.Sigmoid()
        out_real = m(self.init_batch_dict['adv_classification_output'][real_inds]).mean()
        out_syn = m(self.init_batch_dict['adv_classification_output'][syn_inds]).mean()
        #print(out_real - out_syn)
        loss = (out_real - out_syn) * adv_lambda_weighting
        tb_dict['loss_adv'] = loss.item()
        return loss, torch.zeros_like(self.init_batch_dict['data_type']).cuda()



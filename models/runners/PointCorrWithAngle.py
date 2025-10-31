from visualization.visualize_api import visualize_pair_corr, visualize_reconstructions
from data.point_cloud_db.point_cloud_dataset import PointCloudDataset
from models.modules.dgcnn_with_angle import DGCNN_MODULAR
from models.modules.dgcnn import get_graph_feature
from utils.warmup import WarmUpScheduler
from utils.cyclic_scheduler import CyclicLRWithRestarts
import numpy as np
import math
import os
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch_cluster import knn
import pointnet2_ops._ext as _ext
from models.correspondence_utils import get_s_t_neighbors
from models.shape_corr_trainer import ShapeCorrTemplate
from models.metrics.metrics import AccuracyAssumeEye, AccuracyAssumeEyeSoft, uniqueness
from utils import switch_functions
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.dgcnn import DGCNN as non_geo_DGCNN
from utils.argparse_init import str2bool
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from models.data_augment_utils import DataAugment
import scipy.io as sio


class GroupingOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        ctx.for_backwards = (idx, N)
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, N = ctx.for_backwards
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class PointCorrWithAngle(ShapeCorrTemplate):
    def __init__(self, hparams, **kwargs):
        """Stub."""
        super(PointCorrWithAngle, self).__init__(hparams, **kwargs)

        self.updates = 0
        self.decay = 0.9990
        self.methods = ["scale", "noise"]
        # Define 8 discrete angle bins (consistent with orient classifier output) on CPU; moved with module
        angle_bins = (
            torch.arange(8, dtype=torch.float32) * 2 * torch.pi / 8 - torch.pi / 4
        )  # shape (8,)
        self.register_buffer("ANGLE_BINS", angle_bins, persistent=False)
        self.hparams.ANGLE = angle_bins  # passed to submodules if needed
        self.s_model = DGCNN_MODULAR(
            self.hparams, use_inv_features=self.hparams.use_inv_features
        )
        self.t_model = DGCNN_MODULAR(
            self.hparams, use_inv_features=self.hparams.use_inv_features
        )
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.t_model.parameters():
            p.requires_grad = False

        self.chamfer_dist_3d = dist_chamfer_3D.chamfer_3DDist()
        self.accuracy_assume_eye = AccuracyAssumeEye()
        self.accuracy_assume_eye_soft_0p01 = AccuracyAssumeEyeSoft(
            top_k=int(0.01 * self.hparams.num_points)
        )
        self.accuracy_assume_eye_soft_0p05 = AccuracyAssumeEyeSoft(
            top_k=int(0.05 * self.hparams.num_points)
        )
        ## focal_loss
        self.FL_global = FocalLoss(class_num=2, gamma=3)
        self.src_aug = DataAugment(
            operations=["noise"],
            flip_probability=0.5,
            scale_range=[0.95, 1.05],
            noise_variance=0.0001,
        )
        self.tgt_aug = DataAugment(
            operations=["rotate", "noise"],
            flip_probability=0.5,
            scale_range=[0.95, 1.05],
            rotate_nbins=8,
            noise_variance=0.0001,
        )
        self._dv_export_paths = None

    def chamfer_loss(self, pos1, pos2):
        if not pos1.is_cuda:
            pos1 = pos1.cuda()

        if not pos2.is_cuda:
            pos2 = pos2.cuda()

        dist1, dist2, idx1, idx2 = self.chamfer_dist_3d(pos1, pos2)
        loss = torch.mean(dist1) + torch.mean(dist2)

        return loss

    def configure_optimizers(self):
        if self.hparams.warmup:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=0.0, weight_decay=0.0001
            )
            self.scheduler = WarmUpScheduler(
                self.optimizer, [28800, 28800, 0.5], 0.0001
            )
        elif self.hparams.steplr:
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=0.0001, weight_decay=0.0001
            )
            self.scheduler = StepLR(optimizer=self.optimizer, step_size=200, gamma=0.5)
        elif self.hparams.steplr2:
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.slr, weight_decay=self.hparams.swd
            )
            self.scheduler = StepLR(optimizer=self.optimizer, step_size=65, gamma=0.7)
        elif self.hparams.testlr:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.slr * self.hparams.train_batch_size / 4,
                weight_decay=self.hparams.swd,
            )
            self.scheduler = StepLR(optimizer=self.optimizer, step_size=65, gamma=0.7)
        elif self.hparams.cosine:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.slr * self.hparams.train_batch_size / 4,
                weight_decay=self.hparams.swd,
            )
            self.scheduler = CyclicLRWithRestarts(
                optimizer=self.optimizer,
                batch_size=self.hparams.train_batch_size,
                epoch_size=self.hparams.max_epochs,
                restart_period=100,
                t_mult=1.2,
                policy="cosine",
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
            self.scheduler = MultiStepLR(self.optimizer, milestones=[6, 9], gamma=0.1)
        # Lightning (various versions) sometimes rejects bare scheduler objects when returned in a list
        # if they are not standard _LRScheduler subclasses or need additional metadata (interval, etc.).
        # We wrap them in the recommended dict format. For the custom WarmUpScheduler (which also calls
        # optimizer.step inside scheduler.step) we do NOT hand it to Lightning to avoid double stepping.
        from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

        if isinstance(self.scheduler, WarmUpScheduler):
            # We'll manually advance this in training_step by overriding optimizer_step if needed later.
            return {"optimizer": self.optimizer}

        # Standard scheduler path
        sch = self.scheduler
        if isinstance(sch, (ReduceLROnPlateau,)):
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "val_loss",  # placeholder; adjust if a specific metric exists
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if isinstance(sch, (_LRScheduler, CyclicLRWithRestarts, StepLR, MultiStepLR)):
            interval = "epoch"
            # CyclicLRWithRestarts adjusts per batch
            if isinstance(sch, CyclicLRWithRestarts):
                interval = "step"
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "interval": interval,
                    "frequency": 1,
                },
            }

        # Fallback: just return optimizer (prevents ValueError) if scheduler type unexpected
        return {"optimizer": self.optimizer}

    def compute_self_features(self, source, target):
        """In this function, we first perform data augmentation on the point clouds: including rotation and adding noise. Next, we input the augmented point clouds into the network to learn dense features. Finally, we calculate angle loss and domain discriminator loss.

        Args:
            source (tensor): N*3 the source point cloud
            target (tensor): N*3 the target point cloud

        Returns:
            source (dict): the features of source
            target (dict): the features of target
            loss_angle (tensor): (1,) angle loss
            loss_discr (tensor): (1,) domain discriminator loss
        """
        if self.hparams.dataset_name == "shrec":
            return self.shrec_forward(source, target)
        elif self.hparams.dataset_name == "surreal":
            return self.surreal_forward(source, target)
        elif self.hparams.dataset_name == "faust":
            # FAUST consists of human bodies with similar orientation/augmentation policy to SURREAL
            return self.surreal_forward(source, target)
        elif self.hparams.dataset_name == "tosca":
            return self.tosca_forward(source, target)
        elif self.hparams.dataset_name == "smal":
            return self.smal_forward(source, target)

    def shrec_forward(self, source, target):
        """The forward for shrec dataset.

        Args:
            source (tensor): N*3 the source point cloud
            target (tensor): N*3 the target point cloud

        Returns:
            source (dict): the features of source
            target (dict): the features of target
            loss_angle (tensor): (1,) angle loss
            loss_discr (tensor): (1,) domain discriminator loss
        """

        if self.hparams.mode == "train" or self.hparams.mode == "val":
            src_pos_student = self.src_aug(source["pos"])
            tgt_pos_student, rotated_gt = self.tgt_aug(target["pos"])
        else:
            src_pos_student = source["pos"]
            tgt_pos_student = target["pos"]

        # student
        src_out, tgt_out, _, domain_pred_student = self.s_model(
            src_pos_student,
            tgt_pos_student,
            source["neigh_idxs"],
            target["neigh_idxs"],
        )

        # teacher
        src_out_teacher, tgt_out_teacher, _, domain_pred_teacher = self.t_model(
            source["pos"],
            target["pos"],
            source["neigh_idxs"],
            target["neigh_idxs"],
        )

        if self.hparams.mode == "train" or self.hparams.mode == "val":
            # compute angle loss and domain discriminator loss
            _, _, orient_1, domain_pred_target = self.s_model(
                target["pos"],
                tgt_pos_student,
                target["neigh_idxs"],
                target["neigh_idxs"],
            )
            angle_pred = orient_1.reshape((-1, 8))
            rotated_gt = torch.cat((rotated_gt, (10 - rotated_gt) % 8))
            loss_angle = F.cross_entropy(angle_pred, rotated_gt)
            # source domain global
            global_d_pred_S = domain_pred_target
            domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
            source_dloss = 1.0 * self.FL_global(global_d_pred_S, domain_S)
            # target domain global
            global_d_pred_T = domain_pred_student
            domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
            target_dloss = 1.0 * self.FL_global(global_d_pred_T, domain_T)

            loss_discr = source_dloss + target_dloss
        else:
            loss_angle = torch.tensor(0).cuda()
            loss_discr = torch.tensor(0).cuda()

        # student
        source["dense_output_features"] = src_out.transpose(1, 2)
        target["dense_output_features"] = tgt_out.transpose(1, 2)

        # teacher
        source["dense_output_features_teacher"] = src_out_teacher.transpose(1, 2)
        target["dense_output_features_teacher"] = tgt_out_teacher.transpose(1, 2)

        return source, target, loss_angle, loss_discr

    def surreal_forward(self, source, target):
        """The forward for shrec dataset.

        Args:
            source (tensor): N*3 the source point cloud
            target (tensor): N*3 the target point cloud

        Returns:
            source (dict): the features of source
            target (dict): the features of target
            loss_angle (tensor): (1,) angle loss
            loss_discr (tensor): (1,) domain discriminator loss
        """

        if self.hparams.mode == "train" or self.hparams.mode == "val":
            src_pos_student = self.src_aug(source["pos"])
            tgt_pos_student, rotated_gt_0 = self.tgt_aug(target["pos"])
            src_pos_teacher = self.src_aug(source["pos"])
            tgt_pos_teacher, rotated_gt_1 = self.tgt_aug(target["pos"])
        else:
            src_pos_student = source["pos"]
            tgt_pos_student = target["pos"]
            src_pos_teacher = source["pos"]
            tgt_pos_teacher = target["pos"]

        # student
        src_out, tgt_out, orient_0, _ = self.s_model(
            src_pos_student,
            tgt_pos_student,
            source["neigh_idxs"],
            target["neigh_idxs"],
        )

        # teacher
        src_out_teacher, tgt_out_teacher, orient_1, _ = self.t_model(
            src_pos_teacher,
            tgt_pos_teacher,
            source["neigh_idxs"],
            target["neigh_idxs"],
        )

        if self.hparams.mode == "train" or self.hparams.mode == "val":
            # compute angle loss and domain discriminator loss
            _, _, orient_2, _ = self.s_model(
                target["pos"],
                tgt_pos_student,
                target["neigh_idxs"],
                target["neigh_idxs"],
            )
            rotated_gt_2 = (rotated_gt_1 - rotated_gt_0 + 1) % 8
            angle_pred_0 = orient_0.reshape((-1, 8))
            angle_pred_1 = orient_1.reshape((-1, 8))
            angle_pred_2 = orient_2.reshape((-1, 8))
            rotated_gt_0 = torch.cat((rotated_gt_0, (10 - rotated_gt_0) % 8))
            rotated_gt_1 = torch.cat((rotated_gt_1, (10 - rotated_gt_1) % 8))
            rotated_gt_2 = torch.cat((rotated_gt_2, (10 - rotated_gt_2) % 8))
            loss_angle = 0.4 * (
                F.cross_entropy(angle_pred_0, rotated_gt_0)
                + F.cross_entropy(angle_pred_1, rotated_gt_1)
                + F.cross_entropy(angle_pred_2, rotated_gt_2)
            )
            loss_discr = torch.tensor(0).cuda()
        else:
            loss_angle = torch.tensor(0).cuda()
            loss_discr = torch.tensor(0).cuda()

        # student
        source["dense_output_features"] = src_out.transpose(1, 2)
        target["dense_output_features"] = tgt_out.transpose(1, 2)

        # t eacher
        source["dense_output_features_teacher"] = src_out_teacher.transpose(1, 2)
        target["dense_output_features_teacher"] = tgt_out_teacher.transpose(1, 2)

        return source, target, loss_angle, loss_discr

    def tosca_forward(self, source, target):
        """The forward for tosca dataset.

        Args:
            source (tensor): N*3 the source point cloud
            target (tensor): N*3 the target point cloud

        Returns:
            source (dict): the features of source
            target (dict): the features of target
            loss_angle (tensor): (1,) angle loss
            loss_discr (tensor): (1,) domain discriminator loss
        """
        # Coordinate system correction for SMAL and TOSCA
        source["pos"] = self.rotate_point_cloud_for_animal(source["pos"], torch.tensor(-0.5*torch.pi), 'X') #TOSCA
        target["pos"] = self.rotate_point_cloud_for_animal(target["pos"], torch.tensor(-0.5*torch.pi), 'X') #TOSCA

        if self.hparams.mode == "train" or self.hparams.mode == "val":
            src_pos_student = source["pos"]
            tgt_pos_student = target["pos"]
            src_pos_teacher = self.src_aug(source["pos"])
            tgt_pos_teacher, rotated_gt_teacher = self.tgt_aug(target["pos"])
        else:
            src_pos_student = source["pos"]
            tgt_pos_student = target["pos"]
            src_pos_teacher = source["pos"]
            tgt_pos_teacher = target["pos"]

        # student
        src_out, tgt_out, orient_1, domain_pred_student = self.s_model(
            src_pos_student,
            tgt_pos_student,
            source["neigh_idxs"],
            target["neigh_idxs"],
            norm=False,
        )

        # teacher
        src_out_teacher, tgt_out_teacher, orient_2, domain_pred_teacher = self.t_model(
            src_pos_teacher,
            tgt_pos_teacher,
            source["neigh_idxs"],
            target["neigh_idxs"],
            norm=False,
        )

        if self.hparams.mode == "train" or self.hparams.mode == "val":
            # compute angle loss and domain discriminator loss
            _, _, orient_3, domain_pred_target = self.s_model(
                tgt_pos_student,
                tgt_pos_teacher,
                target["neigh_idxs"],
                target["neigh_idxs"],
                norm=False,
            )
            angle_pred_1 = orient_1.reshape((-1, 8))
            angle_pred_2 = orient_2.reshape((-1, 8))
            angle_pred_3 = orient_3.reshape((-1, 8))
            rotated_gt_3 = torch.cat((rotated_gt_teacher, (10 - rotated_gt_teacher) % 8))
            rotated_gt_1 = torch.ones_like(rotated_gt_3)
            rotated_gt_2 = torch.ones_like(rotated_gt_3)
            loss_angle = 0.4 * (
                F.cross_entropy(angle_pred_1, rotated_gt_1)
                + F.cross_entropy(angle_pred_2, rotated_gt_2)
                + F.cross_entropy(angle_pred_3, rotated_gt_3)
            )
            
            # source domain global
            global_d_pred_S = domain_pred_target
            domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
            source_dloss = 1.0 * self.FL_global(global_d_pred_S, domain_S)
            # target domain global
            global_d_pred_T = domain_pred_student
            domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
            target_dloss = 1.0 * self.FL_global(global_d_pred_T, domain_T)

            loss_discr = source_dloss + target_dloss
        else:
            loss_angle = torch.tensor(0).cuda()
            loss_discr = torch.tensor(0).cuda()

        # student
        source["dense_output_features"] = src_out.transpose(1, 2)
        target["dense_output_features"] = tgt_out.transpose(1, 2)

        # teacher
        source["dense_output_features_teacher"] = src_out_teacher.transpose(1, 2)
        target["dense_output_features_teacher"] = tgt_out_teacher.transpose(1, 2)

        return source, target, loss_angle, loss_discr

    def smal_forward(self, source, target):
        """The forward for smal dataset.

        Args:
            source (tensor): N*3 the source point cloud
            target (tensor): N*3 the target point cloud

        Returns:
            source (dict): the features of source
            target (dict): the features of target
            loss_angle (tensor): (1,) angle loss
            loss_discr (tensor): (1,) domain discriminator loss
        """
        # Coordinate system correction for SMAL
        source["pos"] = self.rotate_point_cloud_for_animal(source["pos"], torch.tensor(-0.5*torch.pi), 'Z') #SMAL
        source["pos"] = self.rotate_point_cloud_for_animal(source["pos"], torch.tensor(-0.5*torch.pi), 'X') #SMAL
        target["pos"] = self.rotate_point_cloud_for_animal(target["pos"], torch.tensor(-0.5*torch.pi), 'Z') #SMAL
        target["pos"] = self.rotate_point_cloud_for_animal(target["pos"], torch.tensor(-0.5*torch.pi), 'X') #SMAL

        if self.hparams.mode == "train" or self.hparams.mode == "val":
            src_pos_student = source["pos"]
            tgt_pos_student = target["pos"]
            src_pos_teacher = self.src_aug(source["pos"])
            tgt_pos_teacher, rotated_gt_teacher = self.tgt_aug(target["pos"])
        else:
            src_pos_student = source["pos"]
            tgt_pos_student = target["pos"]
            src_pos_teacher = source["pos"]
            tgt_pos_teacher = target["pos"]

        # student
        src_out, tgt_out, orient_1, domain_pred_student = self.s_model(
            src_pos_student,
            tgt_pos_student,
            source["neigh_idxs"],
            target["neigh_idxs"],
            norm=False,
        )

        # teacher
        src_out_teacher, tgt_out_teacher, orient_2, domain_pred_teacher = self.t_model(
            src_pos_teacher,
            tgt_pos_teacher,
            source["neigh_idxs"],
            target["neigh_idxs"],
            norm=False,
        )

        if self.hparams.mode == "train" or self.hparams.mode == "val":
            # compute angle loss and domain discriminator loss
            _, _, orient_3, domain_pred_target = self.s_model(
                tgt_pos_student,
                tgt_pos_teacher,
                target["neigh_idxs"],
                target["neigh_idxs"],
                norm=False,
            )
            angle_pred_1 = orient_1.reshape((-1, 8))
            angle_pred_2 = orient_2.reshape((-1, 8))
            angle_pred_3 = orient_3.reshape((-1, 8))
            rotated_gt_3 = torch.cat((rotated_gt_teacher, (10 - rotated_gt_teacher) % 8))
            rotated_gt_1 = torch.ones_like(rotated_gt_3)
            rotated_gt_2 = torch.ones_like(rotated_gt_3)
            loss_angle = 0.4 * (
                F.cross_entropy(angle_pred_1, rotated_gt_1)
                + F.cross_entropy(angle_pred_2, rotated_gt_2)
                + F.cross_entropy(angle_pred_3, rotated_gt_3)
            )
            
            # source domain global
            global_d_pred_S = domain_pred_target
            domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
            source_dloss = 1.0 * self.FL_global(global_d_pred_S, domain_S)
            # target domain global
            global_d_pred_T = domain_pred_student
            domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
            target_dloss = 1.0 * self.FL_global(global_d_pred_T, domain_T)

            loss_discr = source_dloss + target_dloss
        else:
            loss_angle = torch.tensor(0).cuda()
            loss_discr = torch.tensor(0).cuda()

        # student
        source["dense_output_features"] = src_out.transpose(1, 2)
        target["dense_output_features"] = tgt_out.transpose(1, 2)

        # teacher
        source["dense_output_features_teacher"] = src_out_teacher.transpose(1, 2)
        target["dense_output_features_teacher"] = tgt_out_teacher.transpose(1, 2)

        return source, target, loss_angle, loss_discr
    
    def rotate_point_cloud_for_animal(self,batch_data, rotation_angle, axis = 'Y'):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
            BxNx3 array, original batch of point clouds
            Return:
            BxNx3 array, rotated batch of point clouds
        """
        rotated_data = torch.zeros(batch_data.shape, dtype=torch.float32).cuda()
        for k in range(batch_data.shape[0]):
            cosval = torch.cos(rotation_angle)
            sinval = torch.sin(rotation_angle)
            if axis == 'X':
                rotation_matrix = torch.tensor([[1, 0, 0],
                                            [0, cosval, sinval],
                                            [0, -sinval, cosval]]).cuda()
            elif axis == 'Y':
                rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]]).cuda()
            elif axis == 'Z':
                rotation_matrix = torch.tensor([[cosval, sinval, 0],
                                            [-sinval, cosval, 0],
                                            [0, 0, 1]]).cuda()

            shape_pc = batch_data[k,:,0:3]
            rotated_data[k,:,0:3] = torch.mm(shape_pc.reshape((-1, 3)), rotation_matrix)

        return rotated_data


    def forward_source_target(self, source, target):
        """In this function, we can get features, similarity, angle loss, and domain discriminator loss

        Args:
            source (tensor): N*3 the source point cloud
            target (tensor): N*3 the target point cloud

        Returns:
            source (dict): the data of source
            target (dict): the data of target
            P_normalized (tensor): N*N the cross similarity of student
            P_non_normalized_teacher (tensor): N*N the cross similarity of teacher
            temperature (tensor): (1,) the temperature
            loss_angle (tensor): (1,) angle loss
            loss_discr (tensor): (1,) domain discriminator loss
        """
        # get features and angle loss
        source, target, loss_angle, loss_discr = self.compute_self_features(
            source, target
        )

        # measure cross similarity of student
        P_non_normalized = switch_functions.measure_similarity(
            self.hparams.similarity_init,
            source["dense_output_features"],
            target["dense_output_features"],
        )

        # measure cross similarity of teacher
        P_non_normalized_teacher = switch_functions.measure_similarity(
            self.hparams.similarity_init,
            source["dense_output_features_teacher"],
            target["dense_output_features_teacher"],
        )

        temperature = None
        P_normalized = P_non_normalized
        P_normalized_teacher = P_non_normalized_teacher

        # cross nearest neighbors and weights
        (
            source["cross_nn_weight"],
            source["cross_nn_sim"],
            source["cross_nn_idx"],
            target["cross_nn_weight"],
            target["cross_nn_sim"],
            target["cross_nn_idx"],
        ) = get_s_t_neighbors(
            self.hparams.k_for_cross_recon,
            P_normalized,
            sim_normalization=self.hparams.sim_normalization,
        )

        # cross reconstruction
        source["cross_recon"], source["cross_recon_hard"] = self.reconstruction(
            source["pos"],
            target["cross_nn_idx"],
            target["cross_nn_weight"],
            self.hparams.k_for_cross_recon,
        )
        target["cross_recon"], target["cross_recon_hard"] = self.reconstruction(
            target["pos"],
            source["cross_nn_idx"],
            source["cross_nn_weight"],
            self.hparams.k_for_cross_recon,
        )

        return (
            source,
            target,
            P_normalized,
            P_normalized_teacher,
            temperature,
            loss_angle,
            loss_discr,
        )

    @staticmethod
    def reconstruction(pos, nn_idx, nn_weight, k):
        nn_pos = get_graph_feature(
            pos.transpose(1, 2),
            k=k,
            idx=nn_idx,
            only_intrinsic="neighs",
            permute_feature=False,
        )
        nn_weighted = nn_pos * nn_weight.unsqueeze(dim=3)
        recon = torch.sum(nn_weighted, dim=2)
        recon_hard = nn_pos[:, :, 0, :]
        return recon, recon_hard

    def forward_shape(self, shape):
        """Compute self similarity and conduct self reconstruction

        Args:
            shape (dict): the relate data of point cloud

        Returns:
            shape (dict): add self-reconstructed point cloud
            P_self (tensor): N*N self similarity of student
            P_self_teacher (tensor): N*N self similarity of teacher
        """
        P_self = switch_functions.measure_similarity(
            self.hparams.similarity_init,
            shape["dense_output_features"],
            shape["dense_output_features"],
        )
        P_self_teacher = switch_functions.measure_similarity(
            self.hparams.similarity_init,
            shape["dense_output_features_teacher"],
            shape["dense_output_features_teacher"],
        )

        # measure self similarity
        nn_idx = (
            shape["neigh_idxs"][:, :, : self.hparams.k_for_self_recon + 1]
            if self.hparams.use_euclidiean_in_self_recon
            else None
        )
        shape["self_nn_weight"], _, shape["self_nn_idx"], _, _, _ = get_s_t_neighbors(
            self.hparams.k_for_self_recon + 1,
            P_self,
            sim_normalization=self.hparams.sim_normalization,
            s_only=True,
            ignore_first=True,
            nn_idx=nn_idx,
        )

        # self reconstruction
        shape["self_recon"], _ = self.reconstruction(
            shape["pos"],
            shape["self_nn_idx"],
            shape["self_nn_weight"],
            self.hparams.k_for_self_recon,
        )

        return shape, P_self, P_self_teacher

    def get_cons_loss(self, P_s, P_t):
        """Compute consistence loss

        Args:
            P_s (tensor): N*N the similarity of student
            P_t (tensor): N*N the similarity of teacher

        Returns:
            tensor: (1,) consistence loss
        """
        cons_loss = F.smooth_l1_loss(P_s, P_t, beta=0.01)
        return cons_loss

    @staticmethod
    def get_neighbor_loss(source, source_neigh_idxs, target_cross_recon, k):
        """Compute neighbour loss

        Args:
            source (tensor): N*3 the pos of source point cloud
            source_neigh_idxs (tensor): N*K the neighbour idxs of source point cloud
            target_cross_recon (tensor): N*3 the pos of reconstruction target point cloud
            k (int): the number of neighbour

        Returns:
            tensor: (1,) the neighbor loss
        """
        # source.shape[1] is the number of points

        if k < source_neigh_idxs.shape[2]:
            neigh_index_for_loss = source_neigh_idxs[:, :, :k]
        else:
            neigh_index_for_loss = source_neigh_idxs

        source_grouped = grouping_operation(
            source.transpose(1, 2).contiguous(), neigh_index_for_loss.int()
        ).permute(0, 2, 3, 1)
        source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(
            source, 2
        )  # remove fist grouped element, as it is the seed point itself
        source_square = torch.sum(source_diff**2, dim=-1)

        target_cr_grouped = grouping_operation(
            target_cross_recon.transpose(1, 2).contiguous(), neigh_index_for_loss.int()
        ).permute(0, 2, 3, 1)
        target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(
            target_cross_recon, 2
        )  # remove fist grouped element, as it is the seed point itself
        target_cr_square = torch.sum(target_cr_diff**2, dim=-1)

        GAUSSIAN_HEAT_KERNEL_T = 8.0
        gaussian_heat_kernel = torch.exp(-source_square / GAUSSIAN_HEAT_KERNEL_T)
        neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

        neighbor_loss = torch.mean(neighbor_loss_per_neigh)

        return neighbor_loss

    def forward(self, data):
        """Main function. In this function, we can get the dense features, similarity, and several losses.

        Args:
            data (dict): the input data

        Returns:
            dict: the outputs which includes features and losses.
        """
        for shape in ["source", "target"]:
            data[shape]["edge_index"] = [
                knn(
                    data[shape]["pos"][i],
                    data[shape]["pos"][i],
                    self.hparams.num_neighs,
                )
                for i in range(data[shape]["pos"].shape[0])
            ]
            data[shape]["neigh_idxs"] = torch.stack(
                [
                    data[shape]["edge_index"][i][1].reshape(
                        data[shape]["pos"].shape[1], -1
                    )
                    for i in range(data[shape]["pos"].shape[0])
                ]
            )

        # dense features, similarity, and cross reconstruction
        (
            data["source"],
            data["target"],
            data["P_normalized"],
            data["P_normalized_teacher"],
            data["temperature"],
            loss_angle,
            loss_discr,
        ) = self.forward_source_target(data["source"], data["target"])

        # angle loss
        self.losses[f"angle_loss"] = self.hparams.angle_lambda * loss_angle
        self.losses[f"discr_loss"] = 1.0 * loss_discr
        # cross reconstruction losses
        self.losses[
            f"source_cross_recon_loss"
        ] = self.hparams.cross_recon_lambda * self.chamfer_loss(
            data["source"]["pos"], data["source"]["cross_recon"]
        )
        self.losses[
            f"target_cross_recon_loss"
        ] = self.hparams.cross_recon_lambda * self.chamfer_loss(
            data["target"]["pos"], data["target"]["cross_recon"]
        )

        # self reconstruction
        if self.hparams.use_self_recon:
            _, P_self_source, P_self_source_teacher = self.forward_shape(data["source"])
            _, P_self_target, P_self_target_teacher = self.forward_shape(data["target"])

            # teacher student consistency losses
            self.losses[f"self_consistency_loss"] = 0.01 * self.get_cons_loss(
                P_self_target, P_self_target_teacher
            )
            self.losses[f"consistency_loss"] = 0.01 * self.get_cons_loss(
                data["P_normalized"], data["P_normalized_teacher"]
            )

            # self reconstruction losses
            data["source"]["self_recon_loss_unscaled"] = self.chamfer_loss(
                data["source"]["pos"], data["source"]["self_recon"]
            )
            data["target"]["self_recon_loss_unscaled"] = self.chamfer_loss(
                data["target"]["pos"], data["target"]["self_recon"]
            )

            self.losses[f"source_self_recon_loss"] = (
                self.hparams.self_recon_lambda
                * data["source"]["self_recon_loss_unscaled"]
            )
            self.losses[f"target_self_recon_loss"] = (
                self.hparams.self_recon_lambda
                * data["target"]["self_recon_loss_unscaled"]
            )

        if self.hparams.compute_neigh_loss and self.hparams.neigh_loss_lambda > 0.0:
            # neighbour losses
            data[f"neigh_loss_fwd_unscaled"] = self.get_neighbor_loss(
                data["source"]["pos"],
                data["source"]["neigh_idxs"],
                data["target"]["cross_recon"],
                self.hparams.k_for_cross_recon,
            )
            data[f"neigh_loss_bac_unscaled"] = self.get_neighbor_loss(
                data["target"]["pos"],
                data["target"]["neigh_idxs"],
                data["source"]["cross_recon"],
                self.hparams.k_for_cross_recon,
            )

            self.losses[f"neigh_loss_fwd"] = (
                self.hparams.neigh_loss_lambda * data[f"neigh_loss_fwd_unscaled"]
            )
            self.losses[f"neigh_loss_bac"] = (
                self.hparams.neigh_loss_lambda * data[f"neigh_loss_bac_unscaled"]
            )

        self.track_metrics(data)

        return data

    @staticmethod
    def is_parallel(model):
        """check if model is in parallel mode."""
        parallel_type = (
            nn.parallel.DataParallel,
            nn.parallel.DistributedDataParallel,
        )
        return isinstance(model, parallel_type)

    def on_train_batch_end(self, trainer, pl_module, outputs, unused=0):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay * (1 - math.exp(-self.updates / 20000))

            msd = (
                self.s_model.module.state_dict()
                if self.is_parallel(self.s_model)
                else self.s_model.state_dict()
            )  # model state_dict
            for k, v in self.t_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def test_step(self, test_batch, batch_idx):
        """For eval and test

        Args:
            test_batch (dict): the input data
            batch_idx (int): the idx of batch

        Returns:
            bool: whether the process is successful
        """
        self.batch = test_batch
        self.hparams.mode = "test"
        self.hparams.batch_idx = batch_idx

        label, pinput1, input2, ratio_list, soft_labels = self.extract_labels_for_test(
            test_batch
        )

        source_name = self._resolve_shape_name(self.batch["source"], "source")
        target_name = self._resolve_shape_name(self.batch["target"], "target")
        source = {
            "pos": pinput1,
            "id": self.batch["source"]["id"],
            "name": source_name,
            "orig_pos": self.batch["source"].get("orig_pos"),
            "subsample_idx": self.batch["source"].get("subsample_idx"),
        }
        target = {
            "pos": input2,
            "id": self.batch["target"]["id"],
            "name": target_name,
            "orig_pos": self.batch["target"].get("orig_pos"),
            "subsample_idx": self.batch["target"].get("subsample_idx"),
        }
        batch = {"source": source, "target": target}
        batch = self(batch)
        p = batch["P_normalized_teacher"].clone()
        self._export_dv_outputs(batch)

        if self.hparams.use_dualsoftmax_loss:
            temp = 0.0002
            p = (
                p * F.softmax(p / temp, dim=0) * len(p)
            )  # With an appropriate temperature parameter, the model achieves higher performance
            p = F.log_softmax(p, dim=-1)

        _ = self.compute_acc(
            label,
            ratio_list,
            soft_labels,
            p,
            input2,
            track_dict=self.tracks,
            hparams=self.hparams,
        )

        self.log_test_step()
        if self.vis_iter():
            self.visualize(batch, mode="test")

        return True

    def _export_dv_outputs(self, batch):
        if not getattr(self.hparams, "save_dv_outputs", True):
            return

        export_paths = self._get_dv_export_paths()
        source_entry = batch.get("source", {})
        target_entry = batch.get("target", {})
        source_name = self._resolve_shape_name(source_entry, "source")
        target_name = self._resolve_shape_name(target_entry, "target")

        src_student = source_entry.get("dense_output_features")
        tgt_student = target_entry.get("dense_output_features")
        src_teacher = source_entry.get("dense_output_features_teacher")
        tgt_teacher = target_entry.get("dense_output_features_teacher")

        if src_student is None or tgt_student is None:
            return

        if src_teacher is None:
            src_teacher = src_student
        if tgt_teacher is None:
            tgt_teacher = tgt_student

        with torch.no_grad():
            src_teacher_feat = src_teacher.detach().squeeze(0)
            tgt_teacher_feat = tgt_teacher.detach().squeeze(0)
            src_teacher_feat = src_teacher_feat.to(
                dtype=torch.float32, memory_format=torch.contiguous_format
            )
            tgt_teacher_feat = tgt_teacher_feat.to(
                dtype=torch.float32, memory_format=torch.contiguous_format
            )
            dist12 = torch.cdist(
                src_teacher_feat,
                tgt_teacher_feat,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            T12 = torch.argmin(dist12, dim=1).cpu().numpy().astype(np.int64)
            T21 = torch.argmin(dist12, dim=0).cpu().numpy().astype(np.int64)

        T12_full = self._expand_correspondence(T12, source_entry, target_entry)
        T21_full = self._expand_correspondence(T21, target_entry, source_entry)

        np.savetxt(
            os.path.join(export_paths["T"], f"T_{source_name}_{target_name}.txt"),
            (T12_full + 1).reshape(-1, 1),
            fmt="%i",
        )
        np.savetxt(
            os.path.join(export_paths["T"], f"T_{target_name}_{source_name}.txt"),
            (T21_full + 1).reshape(-1, 1),
            fmt="%i",
        )

        src_student_subset = src_student.detach().squeeze(0).cpu().numpy()
        tgt_student_subset = tgt_student.detach().squeeze(0).cpu().numpy()
        src_teacher_subset = src_teacher_feat.cpu().numpy()
        tgt_teacher_subset = tgt_teacher_feat.cpu().numpy()

        src_teacher_np = self._expand_features(src_teacher_subset, source_entry)
        tgt_teacher_np = self._expand_features(tgt_teacher_subset, target_entry)
        src_student_feat = self._expand_features(src_student_subset, source_entry)
        tgt_student_feat = self._expand_features(tgt_student_subset, target_entry)

        feature_payload_src = {
            "uphi": src_teacher_np,
            "uphi_student": src_student_feat,
        }
        feature_payload_tgt = {
            "uphi": tgt_teacher_np,
            "uphi_student": tgt_student_feat,
        }
        feature_payload_src["uphi_teacher"] = src_teacher_np
        feature_payload_tgt["uphi_teacher"] = tgt_teacher_np

        sio.savemat(
            os.path.join(export_paths["feature"], f"usefeature_{source_name}.mat"),
            feature_payload_src,
        )
        sio.savemat(
            os.path.join(export_paths["feature"], f"usefeature_{target_name}.mat"),
            feature_payload_tgt,
        )

    def _expand_correspondence(self, subset_corr, src_entry, tgt_entry):
        subset_corr = np.asarray(subset_corr, dtype=np.int64)
        src_idx = self._tensor_to_numpy(src_entry.get("subsample_idx"))
        tgt_idx = self._tensor_to_numpy(tgt_entry.get("subsample_idx"))
        src_full_pos = self._tensor_to_numpy(src_entry.get("orig_pos"))
        if src_idx is None or tgt_idx is None or src_full_pos is None:
            return subset_corr

        src_idx = src_idx.astype(np.int64)
        tgt_idx = tgt_idx.astype(np.int64)
        if subset_corr.shape[0] != src_idx.shape[0]:
            min_len = min(subset_corr.shape[0], src_idx.shape[0])
            subset_corr = subset_corr[:min_len]
            src_idx = src_idx[:min_len]

        num_src_full = src_full_pos.shape[0]
        full_corr = np.empty(num_src_full, dtype=np.int64)
        full_corr.fill(-1)

        subset_corr_clipped = np.clip(subset_corr, 0, max(len(tgt_idx) - 1, 0))
        mapped_targets = tgt_idx[subset_corr_clipped]
        if mapped_targets.size == 0:
            return np.zeros(num_src_full, dtype=np.int64)
        full_corr[src_idx] = mapped_targets

        missing_src = np.setdiff1d(
            np.arange(num_src_full, dtype=np.int64),
            src_idx,
            assume_unique=True,
        )
        if missing_src.size > 0 and src_idx.size > 0:
            subset_pos = src_full_pos[src_idx]
            missing_pos = src_full_pos[missing_src]
            diff = missing_pos[:, None, :] - subset_pos[None, :, :]
            dist = np.sum(diff * diff, axis=2)
            nn = np.argmin(dist, axis=1)
            full_corr[missing_src] = mapped_targets[nn]

        # For any remaining -1 (can happen if tgt subset empty), fall back to nearest valid target or zero
        if (full_corr < 0).any():
            fallback = mapped_targets[0] if mapped_targets.size > 0 else 0
            full_corr[full_corr < 0] = fallback

        return full_corr

    def _expand_features(self, subset_feats, entry):
        subset_feats = np.asarray(subset_feats)
        full_pos = self._tensor_to_numpy(entry.get("orig_pos"))
        subset_idx = self._tensor_to_numpy(entry.get("subsample_idx"))
        if full_pos is None or subset_idx is None:
            return subset_feats

        if subset_feats.ndim != 2 or subset_feats.size == 0:
            return subset_feats

        subset_idx = subset_idx.astype(np.int64)
        num_full = full_pos.shape[0]
        if subset_feats.shape[0] == num_full:
            return subset_feats

        full_array = np.zeros((num_full, subset_feats.shape[1]), dtype=subset_feats.dtype)
        valid_len = min(subset_feats.shape[0], subset_idx.shape[0])
        full_array[subset_idx[:valid_len]] = subset_feats[:valid_len]

        missing = np.setdiff1d(
            np.arange(num_full, dtype=np.int64),
            subset_idx[:valid_len],
            assume_unique=True,
        )
        if missing.size > 0 and valid_len > 0:
            subset_pos = full_pos[subset_idx[:valid_len]]
            missing_pos = full_pos[missing]
            diff = missing_pos[:, None, :] - subset_pos[None, :, :]
            dist = np.sum(diff * diff, axis=2)
            nn = np.argmin(dist, axis=1)
            full_array[missing] = subset_feats[nn]
        return full_array

    def _tensor_to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            arr = value
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            return None

        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr.squeeze(axis=0)
        if arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr.squeeze(axis=-1)
        return arr

    def _get_dv_export_paths(self):
        if self._dv_export_paths is not None:
            return self._dv_export_paths

        base_dir = getattr(self.hparams, "dv_export_dir", None)
        if base_dir is None:
            run_alias = (
                getattr(self.hparams, "exp_name", None)
                or getattr(self.hparams, "experiment_name", None)
                or self.__class__.__name__
            )
            dataset = getattr(self.hparams, "dataset_name", "dataset")
            base_dir = os.path.join(
                os.getcwd(),
                "result",
                f"{run_alias}_{dataset}",
            )

        paths = {
            "base": base_dir,
            "T": os.path.join(base_dir, "T"),
            "feature": os.path.join(base_dir, "feature"),
        }
        os.makedirs(paths["T"], exist_ok=True)
        os.makedirs(paths["feature"], exist_ok=True)
        self._dv_export_paths = paths
        return self._dv_export_paths

    def _resolve_shape_name(self, entry, fallback_prefix):
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, (list, tuple)):
            name = name[0] if len(name) > 0 else None
        if isinstance(name, torch.Tensor):
            if name.numel() == 1:
                name = name.item()
            else:
                name = name.squeeze().cpu().tolist()
        if isinstance(name, np.ndarray):
            if name.size == 1:
                name = name.item()
        if name is None:
            sid = self._resolve_shape_id(entry if isinstance(entry, dict) else {})
            if isinstance(sid, (int, np.integer)):
                name = f"{fallback_prefix}_{int(sid)}"
            else:
                name = fallback_prefix
        if isinstance(name, (int, float, np.integer)):
            name = f"{fallback_prefix}_{int(name)}"
        safe_name = str(name)
        safe_name = safe_name.replace("\\", "_").replace("/", "_")
        safe_name = safe_name.replace(" ", "_")
        return safe_name

    def _resolve_shape_id(self, entry):
        sid = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(sid, torch.Tensor):
            if sid.numel() == 1:
                return int(sid.item())
            sid = sid.squeeze().cpu().numpy()
        if isinstance(sid, np.ndarray):
            if sid.size == 1:
                return int(sid.item())
        if isinstance(sid, (list, tuple)) and len(sid) > 0:
            return int(sid[0])
        if isinstance(sid, (int, float, np.integer)):
            return int(sid)
        return None

    def visualize(self, batch, mode="train"):
        visualize_pair_corr(self, batch, mode=mode)
        visualize_reconstructions(self, batch, mode=mode)

    @staticmethod
    def add_model_specific_args(parent_parser, task_name, dataset_name):
        """Update the args

        Args:
            parent_parser (dict): the parent args
            task_name (str): the name of the task
            dataset_name (str): the name of the dataset

        Returns:
            dict: the updated args
        """
        parser = ShapeCorrTemplate.add_model_specific_args(
            parent_parser, task_name, dataset_name, is_lowest_leaf=False
        )
        parser = non_geo_DGCNN.add_model_specific_args(
            parser, task_name, dataset_name, is_lowest_leaf=True
        )
        parser = PointCloudDataset.add_dataset_specific_args(
            parser, task_name, dataset_name, is_lowest_leaf=False
        )
        parser.add_argument(
            "--k_for_cross_recon",
            default=10,
            type=int,
            help="number of neighbors for cross reconstruction",
        )

        parser.add_argument(
            "--use_self_recon",
            nargs="?",
            default=True,
            type=str2bool,
            const=True,
            help="whether to use self reconstruction",
        )
        parser.add_argument(
            "--k_for_self_recon",
            default=10,
            type=int,
            help="number of neighbors for self reconstruction",
        )
        parser.add_argument(
            "--self_recon_lambda",
            type=float,
            default=10.0,
            help="weight for self reconstruction loss",
        )
        parser.add_argument(
            "--cross_recon_lambda",
            type=float,
            default=1.0,
            help="weight for cross reconstruction loss",
        )
        parser.add_argument(
            "--consistency_lambda",
            type=float,
            default=0.1,
            help="weight for consistency loss",
        )
        parser.add_argument(
            "--angle_lambda", type=float, default=1, help="weight for angle loss"
        )
        parser.add_argument(
            "--save_dv_outputs",
            nargs="?",
            default=True,
            type=str2bool,
            const=True,
            help="whether to export DV-Matcher style T/features during test",
        )
        parser.add_argument(
            "--dv_export_dir",
            type=str,
            default=None,
            help="custom directory for DV-style outputs",
        )

        parser.add_argument(
            "--compute_perm_loss",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to compute permutation loss",
        )
        parser.add_argument(
            "--perm_loss_lambda",
            type=float,
            default=1.0,
            help="weight for permutation loss",
        )

        parser.add_argument(
            "--optimize_pos",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to compute neighbor smoothness loss",
        )
        parser.add_argument(
            "--compute_neigh_loss",
            nargs="?",
            default=True,
            type=str2bool,
            const=True,
            help="whether to compute neighbor smoothness loss",
        )
        parser.add_argument(
            "--neigh_loss_lambda",
            type=float,
            default=1.0,
            help="weight for neighbor smoothness loss",
        )
        parser.add_argument(
            "--num_angles",
            type=int,
            default=100,
        )

        parser.add_argument(
            "--use_euclidiean_in_self_recon",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use self reconstruction",
        )
        parser.add_argument(
            "--use_all_neighs_for_cross_reco",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use self reconstruction",
        )
        parser.add_argument(
            "--use_all_neighs_for_self_reco",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use self reconstruction",
        )

        """
        PreEncoder-related args
        """
        parser.add_argument(
            "--use_preenc",
            nargs="?",
            default=True,
            type=str2bool,
            const=False,
            help="whether to use DGCNN pre-encoder",
        )

        """
        Transformer-related args
        """
        parser.add_argument(
            "--enc_type", type=str, default="vanilla", help="attention mechanism type"
        )
        parser.add_argument(
            "--d_embed", type=int, default=512, help="transformer embedding dim"
        )
        parser.add_argument(
            "--nhead", type=int, default=8, help="transformer multi-head number"
        )
        parser.add_argument(
            "--d_feedforward", type=int, default=1024, help="feed forward dim"
        )
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
        parser.add_argument(
            "--transformer_act",
            type=str,
            default="relu",
            help="activation function in transformer",
        )
        parser.add_argument(
            "--pre_norm",
            nargs="?",
            default=True,
            type=str2bool,
            const=False,
            help="whether to use prenormalization",
        )
        parser.add_argument(
            "--sa_val_has_pos_emb",
            nargs="?",
            default=True,
            type=str2bool,
            const=False,
            help="position embedding in self-attention",
        )
        parser.add_argument(
            "--ca_val_has_pos_emb",
            nargs="?",
            default=True,
            type=str2bool,
            const=False,
            help="position embedding in cross-attention",
        )
        parser.add_argument(
            "--attention_type",
            type=str,
            default="dot_prod",
            help="attention mechanism type",
        )
        parser.add_argument(
            "--num_encoder_layers",
            type=int,
            default=6,
            help="the number of transformer encoder layers",
        )
        parser.add_argument(
            "--transformer_encoder_has_pos_emb",
            nargs="?",
            default=True,
            type=str2bool,
            const=False,
            help="whether to use position embedding in transformer encoder",
        )
        parser.add_argument(
            "--warmup",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use warmup",
        )
        parser.add_argument(
            "--steplr",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use StepLR",
        )
        parser.add_argument(
            "--steplr2",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use StepLR2",
        )
        parser.add_argument(
            "--testlr",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use test lr",
        )
        parser.add_argument(
            "--cosine",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use test lr",
        )
        parser.add_argument(
            "--slr", type=float, default=5e-4, help="steplr learning rate"
        )
        parser.add_argument(
            "--swd", default=5e-4, type=float, help="steplr2 weight decay"
        )
        parser.add_argument(
            "--layer_list",
            type=list,
            default=["s", "c", "s", "c", "s", "c", "s", "c"],
            help="encoder layer list",
        )

        """
        Shape Selective Whitening Loss-related args
        """
        parser.add_argument(
            "--old_ssw",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use old shape whitening loss(similar to stereowhiteningloss)",
        )
        parser.add_argument(
            "--compute_ssw_loss",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to compute shape selective whitening loss",
        )
        parser.add_argument(
            "--ssw_loss_lambda", type=float, default=3.0, help="weight for SSW loss"
        )

        """
        Dual Softmax Loss-related args
        """
        parser.add_argument(
            "--use_dualsoftmax_loss",
            nargs="?",
            default=False,
            type=str2bool,
            const=True,
            help="whether to use dual softmax loss",
        )

        parser.add_argument(
            "--compute_geodesic_error",
            nargs="?",
            default=True,
            type=str2bool,
            const=True,
            help="whether to compute geodesic error during testing (may be computationally expensive)",
        )

        parser.set_defaults(
            optimizer="adam",
            lr=0.0003,
            weight_decay=5e-4,
            max_epochs=300,
            accumulate_grad_batches=2,
            latent_dim=768,
            bb_size=24,
            num_neighs=27,
            val_vis_interval=20,
            test_vis_interval=20,
        )

        return parser

    def track_metrics(self, data):
        """Track metrics

        Args:
            data (dict): data of metrics
        """

        self.tracks[f"source_cross_recon_error"] = self.chamfer_loss(
            data["source"]["pos"], data["source"]["cross_recon_hard"]
        )
        self.tracks[f"target_cross_recon_error"] = self.chamfer_loss(
            data["target"]["pos"], data["target"]["cross_recon_hard"]
        )
        _, P_self_target, P_self_target_teacher = self.forward_shape(data["target"])
        self.tracks[f"consistency_loss"] = self.get_cons_loss(
            data["P_normalized"], data["P_normalized_teacher"]
        )
        self.tracks[f"self_consistency_loss"] = self.get_cons_loss(
            P_self_target, P_self_target_teacher
        )
        if self.hparams.use_self_recon:
            self.tracks[f"source_self_recon_loss_unscaled"] = data["source"][
                "self_recon_loss_unscaled"
            ]
            self.tracks[f"target_self_recon_loss_unscaled"] = data["target"][
                "self_recon_loss_unscaled"
            ]

        if self.hparams.compute_neigh_loss and self.hparams.neigh_loss_lambda > 0.0:
            self.tracks[f"neigh_loss_fwd_unscaled"] = data[f"neigh_loss_fwd_unscaled"]
            self.tracks[f"neigh_loss_bac_unscaled"] = data[f"neigh_loss_bac_unscaled"]

        # nearest neighbors hit accuracy
        source_pred = data["P_normalized"].argmax(dim=2)
        target_neigh_idxs = data["target"]["neigh_idxs"]

        target_pred = data["P_normalized"].argmax(dim=1)
        source_neigh_idxs = data["source"]["neigh_idxs"]

        # uniqueness (number of unique predictions)
        self.tracks[f"uniqueness_fwd"] = uniqueness(source_pred)
        self.tracks[f"uniqueness_bac"] = uniqueness(target_pred)


class FocalLoss(nn.Module):
    def __init__(
        self,
        class_num,
        alpha=None,
        gamma=2,
        size_average=True,
        sigmoid=False,
        reduce=True,
    ):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce

    def forward(self, inputs, targets, global_weight=None):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            # F.softmax(inputs)
            if targets == 0:
                probs = 1 - P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        else:
            # inputs = F.sigmoid(inputs)
            P = F.softmax(inputs, dim=-1)
            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.0)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]
            probs = (P * class_mask).sum(1).view(-1, 1)
            log_p = probs.log()
            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if not self.reduce:
            return batch_loss
        if self.size_average:
            if global_weight is not None:
                global_weight = global_weight.view(-1, 1)
                batch_loss = batch_loss * global_weight
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
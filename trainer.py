import os
import time
import timm
import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from utils.utils import *
from utils.visualizer import Visualizer, denormalization
from datasets.mvtec import MVTecDataset, MVTEC_CLASS_NAMES
from datasets.visa import VisADataset, VISA_CLASS_NAMES
from datasets.smd import SMDDataset, SMD_CLASS_NAMES
from datasets.psm import PSMDataset, PSM_CLASS_NAMES
from datasets.swat import SWATDataset, SWAT_CLASS_NAMES
from datasets.msl import MSLDataset, MSL_CLASS_NAMES
from datasets.smap import SMAPDataset, SMAP_CLASS_NAMES
from models.model import RLR
from models.decoder import Decoder
from evaluation.basic_metrics import basic_metricor

from timm.models.resnet import _cfg as res_cfg
from timm.models.efficientnet import _cfg as efn_cfg


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.class_names = MVTEC_CLASS_NAMES
        dataset_class = MVTecDataset
        if args.dataset == 'visa':
            self.class_names = VISA_CLASS_NAMES
            dataset_class = VisADataset
        if args.dataset == 'smd':
            self.class_names = SMD_CLASS_NAMES
            dataset_class = SMDDataset
        if args.dataset == 'psm':   
            self.class_names = PSM_CLASS_NAMES
            dataset_class = PSMDataset
        if args.dataset == 'swat':
            self.class_names = SWAT_CLASS_NAMES
            dataset_class = SWATDataset
        if args.dataset == 'msl':
            self.class_names = MSL_CLASS_NAMES
            dataset_class = MSLDataset
        if args.dataset == 'smap':
            self.class_names = SMAP_CLASS_NAMES
            dataset_class = SMAPDataset
        
        train_dataset = dataset_class(args, is_train=True)
        print('Train set size: ', len(train_dataset))
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)

        self.test_loaders = {}
        tsize = 0
        for c in self.class_names:
            test_dataset = dataset_class(args, is_train=False, class_name=c)
            tsize += len(test_dataset)
            self.test_loaders[c] = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        print('Test set size: ', tsize)

        self.start_epoch = 0
        self.best_img_f1 = 0.0
        self.best_pix_f1 = 0.0
        self.best_img_f1s = []
        self.best_pix_f1s = []
        self.build_model()
        self.l2_criterion = nn.MSELoss()
        self.cos_criterion = nn.CosineSimilarity(dim=-1)

    def build_model(self):
        if 'efficientnet' in self.args.backbone_arch:
            config = efn_cfg(url='', file=f'{self.args.root_path}/pretrained/tf_efficientnet_b6_aa-80ba17e4.pth')
        elif 'resnet50' in self.args.backbone_arch:
            config = res_cfg(url='', file=f'{self.args.root_path}/pretrained/wide_resnet50_racm-8234f177.pth')
        encoder = timm.create_model(
            self.args.backbone_arch,
            features_only=True,
            # pretrained_cfg=config, 
            out_indices=self.args.out_indices,
            pretrained=True
        )
        self.encoder = encoder.to(self.args.device).eval()
        
        feat_dims = encoder.feature_info.channels()
        print("Feature Dimensions:", feat_dims)
        
        models = []
        self.seq_lens = []
        self.ws = []
        hid_dim = []
        for i in self.args.out_indices:
            ws = self.args.inp_size // (2 ** (i + 1))
            self.ws.append(ws)
            self.seq_lens.append(ws ** 2)
            hid_dim.append(128 * (2 ** (i - 1)))
        for seq_len, in_channels, d_model in zip(self.seq_lens, feat_dims, hid_dim):
            model = RLR(
                seq_len=seq_len,
                in_channels=in_channels,
                out_channels=in_channels,
                d_model=d_model,
                n_heads=8,
                args=self.args
            )
            models.append(model.to(self.args.device))
        self.models = models
        checkpoint = None
        path = os.path.join(self.args.save_path, self.args.save_prefix, 'latest.pth')
        if os.path.exists(path):
            print('Resume..........')
            checkpoint = torch.load(path)
            state_dict = checkpoint['state_dict']
            for i, model in enumerate(self.models):
                model.load_state_dict(state_dict[i])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_img_f1 = checkpoint['best_img_f1']
            self.best_pix_f1 = checkpoint['best_pix_f1']
            self.best_img_f1s = checkpoint['best_img_f1s']
            self.best_pix_f1s = checkpoint['best_pix_f1s']
        print('Creating Models...Done')
        params_m = list(models[0].parameters())
        for l in range(1, self.args.feature_levels):
            params_m += list(models[l].parameters())
        self.optimizer = torch.optim.Adam(params_m, lr=self.args.lr)

    def add_jitter(self, feature, scale, prob, device, with_mask):
        batch_size, num_tokens, dim_channel = feature.shape
        if with_mask:
            num_mask = int(prob * num_tokens)
            shuffle_indices = torch.rand(batch_size, num_tokens, device=device).argsort()
            mask_ind = shuffle_indices[:, :num_mask]
            batch_ind = torch.arange(batch_size, device=device).unsqueeze(-1)
            feature_new = feature.detach().clone()
            feature_new[batch_ind, mask_ind] = 0
            feature = feature_new
        else:
            num_not_jitter = int((1 - prob) * num_tokens)
            shuffle_indices = torch.rand(batch_size, num_tokens, device=device).argsort()
            not_jitter_ind = shuffle_indices[:, :num_not_jitter]
            batch_ind = torch.arange(batch_size, device=device).unsqueeze(-1)
            feature_norms = (feature.norm(dim=2).unsqueeze(2) / dim_channel)
            jitter = torch.randn((batch_size, num_tokens, dim_channel)).to(device)
            jitter = jitter * feature_norms * scale
            jitter[batch_ind, not_jitter_ind] = 0
            feature = feature + jitter
        return feature

    def train(self):
        path = os.path.join(self.args.save_path, self.args.save_prefix)
        if not os.path.exists(path):
            os.makedirs(path)

        start_time = time.time()
        train_steps = len(self.train_loader)
        best_img_f1, best_pix_f1 = self.best_img_f1, self.best_pix_f1
        best_img_f1s, best_pix_f1s = self.best_img_f1s, self.best_pix_f1s

        for epoch in range(self.start_epoch, self.args.num_epochs):
            print("======================TRAIN MODE======================")
            iter_count = 0
            loss_list = []

            epoch_time = time.time()
            for model in self.models:
                model.train()
            for i, (images, _, _, _, _) in enumerate(self.train_loader):
                iter_count += 1
                images = images.float().to(self.args.device)  # (N, 3, H, W)
                
                with torch.no_grad():
                    features = self.encoder(images)
                
                for fl in range(self.args.feature_levels):
                    if self.args.no_avg:
                        input = features[fl]
                    else:
                        m = torch.nn.AvgPool2d(3, 1, 1)
                        input = m(features[fl])
                    N, D, _, _ = input.shape
                    input = input.permute(0, 2, 3, 1).reshape(N, -1, D)
                    if self.args.feature_jitter > 0 or self.args.with_mask:
                        input_fj = self.add_jitter(
                            input, self.args.feature_jitter, self.args.noise_prob, self.args.device, self.args.with_mask
                        )
                    else:
                        input_fj = input

                    # output: reconstructed features, (N, L, dim)
                    model = self.models[fl]
                    output = model(input_fj)

                    loss = self.l2_criterion(output, input) + torch.mean(1 - self.cos_criterion(output, input)) # mse + cosine
                    if self.args.ref_loss and self.args.ref_len > 1:
                        seq_len = self.seq_lens[fl]
                        ll = (self.args.ref_len - 1) * self.args.ref_len
                        for i in range(self.args.ref_len - 1):
                            si, ei = i * seq_len, (i + 1) * seq_len
                            for j in range(i + 1, self.args.ref_len):
                                sj, ej = j * seq_len, (j + 1) * seq_len
                                loss += torch.mean(self.cos_criterion(model.ref_mca[:, si:ei, :], model.ref_mca[:, sj:ej, :])) / ll

                    loss_list.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
            speed = (time.time() - start_time) / iter_count
            left_time = speed * ((self.args.num_epochs - epoch) * train_steps - i)
            print("Epoch: {} cost time: {}s | speed: {:.4f}s/iter | left time: {:.4f}s".format(epoch + 1, time.time() - epoch_time, speed, left_time))
            iter_count = 0
            start_time = time.time()

            print("Epoch: {0}, Steps: {1} | Rec Loss: {2:.7f}".format(epoch + 1, train_steps, np.average(loss_list)))

            img_f1, pix_f1, img_f1s, pix_f1s = self.test(vis=False)

            for c, i, p in zip(self.class_names, img_f1s, pix_f1s):
                print(f"class: {c} | img f1: {i} | pix f1: {p}")
            print(f"Avg | img f1: {img_f1} | pix f1: {pix_f1}")
            
            state = {
                'state_dict': [model.state_dict() for model in self.models],
                'epoch': epoch,
                'best_img_f1': best_img_f1,
                'best_pix_f1': best_pix_f1,
                'best_img_f1s': best_img_f1s,
                'best_pix_f1s': best_pix_f1s,
            }
            if img_f1 > best_img_f1:
                best_img_f1 = img_f1
                best_img_f1s = img_f1s
                state['best_img_f1'] = best_img_f1
                state['best_img_f1s'] = best_img_f1s
                torch.save(state, os.path.join(path, 'best-img.pth'))
            if pix_f1 > best_pix_f1:
                best_pix_f1 = pix_f1
                best_pix_f1s = pix_f1s
                state['best_pix_f1c'] = best_pix_f1
                state['best_pix_f1s'] = best_pix_f1s
                torch.save(state, os.path.join(path, 'best-pix.pth'))
            torch.save(state, os.path.join(path, 'latest.pth'))
        
        return best_img_f1, best_pix_f1, best_img_f1s, best_pix_f1s

    def test(self, vis=False, checkpoint_path=None):
        print("======================TEST MODE======================")
        if checkpoint_path is not None:
            checkpoint = torch.load(os.path.join(checkpoint_path, self.args.save_prefix, 'best-img.pth'))
            state_dict = checkpoint['state_dict']
            for i, model in enumerate(self.models):
                model.load_state_dict(state_dict[i])
        for model in self.models:
            model.eval()
        img_f1s, pix_f1s = [], []
        for class_name in self.class_names:
            img_f1, pix_f1 = self.test_separate(class_name, vis)
            img_f1s.append(img_f1)
            pix_f1s.append(pix_f1)
        return np.mean(img_f1s), np.mean(pix_f1s), img_f1s, pix_f1s
    
    def test_separate(self, class_name, vis=False):
        l2_criterion = nn.MSELoss(reduction='none')
        cos_criterion = nn.CosineSimilarity(dim=-1)

        decoder = None
        if self.args.with_decoder:
            decoder = Decoder(self.args.feature_levels).to(self.args.device)
            if self.args.no_avg:
                decoder_path = f'{self.args.root_path}/decoder_noavg'
            else:
                decoder_path = f'{self.args.root_path}/decoder'
            if self.args.feature_levels == 3:
                decoder_path += '_fl3'
            checkpoint = torch.load(f'{decoder_path}/{class_name}/best-train.pth')
            state_dict = checkpoint['state_dict']
            decoder.load_state_dict(state_dict)
            decoder.eval()
            if vis and self.models[0].ref_mca is not None:
                refs = []
                for i in range(self.args.feature_levels):
                    ref = self.models[i].ref_mca
                    _, _, D = ref.shape
                    H = W = self.ws[i]
                    refs.append(ref.permute(0, 2, 1).reshape(1, D, H, W))
                ref_out = decoder(refs)[0].detach().cpu().numpy()
                ref_out = denormalization(ref_out)
                ref_path = f'{self.args.root_path}/vis_results/{self.args.save_prefix}/{class_name}'
                os.makedirs(ref_path, exist_ok=True)
                cv2.imwrite(f'{ref_path}/ref_mca.png', ref_out)

        scores_list = [list() for _ in range(self.args.feature_levels)]
        decode_list = []
        test_imgs, gt_label_list, gt_mask_list, file_names, img_anomalies = [], [], [], [], []
        for i, (image, label, mask, file_name, img_anomaly) in enumerate(self.test_loaders[class_name]):
            test_imgs.append(image.cpu().numpy())
            gt_label_list.extend(label)
            gt_mask_list.extend(mask.numpy())
            file_names.extend(file_name)
            img_anomalies.extend(img_anomaly)
            
            image = image.float().to(self.args.device)

            with torch.no_grad():
                features = self.encoder(image)
                inputs = []
                for fl in range(self.args.feature_levels):
                    if self.args.no_avg:
                        input = features[fl]
                    else:
                        m = torch.nn.AvgPool2d(3, 1, 1)
                        input = m(features[fl])
                    N, D, H, W = input.shape
                    input = input.permute(0, 2, 3, 1).reshape(N, -1, D)
                    
                    model = self.models[fl]
                    output = model(input)
                    inputs.append(output.permute(0, 2, 1).reshape(N, D, H, W))

                    score = torch.mean(l2_criterion(input, output), dim=-1) + 1 - cos_criterion(input, output)

                    score = score.detach()  # (N, L)
                    score = score.reshape(score.shape[0], self.ws[fl], self.ws[fl])
                    score = F.interpolate(
                        score.unsqueeze(1),
                        size=self.args.inp_size,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(1).cpu().numpy()
                    scores_list[fl].append(score)
                if decoder is not None:
                    decode_out = decoder(inputs)
                    decode_list.append(decode_out.detach().cpu().numpy())
            
        lvl_scores = []
        for l in range(self.args.feature_levels):
            lvl_score = np.concatenate(scores_list[l], axis=0)  # (N, 256, 256)
            lvl_scores.append(lvl_score)
            
        scores = np.zeros_like(lvl_scores[0])
        for l in range(self.args.feature_levels):
            scores += lvl_scores[l]
        scores = scores / self.args.feature_levels

        for i in range(scores.shape[0]):
            scores[i] = gaussian_filter(scores[i], sigma=4)

        # ======pix evaluation======
        threshold = np.percentile(scores, 99)
        pred_label = np.asarray(scores >= threshold, dtype=bool)
        pred_label_pix = np.max(pred_label, axis=1)

        gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
        gt_mask_pix = np.max(gt_mask, axis=1)

        evaluator = basic_metricor()
        aff_p_pix, aff_r_pix, aff_f1_pix = evaluator.metric_Affiliation(gt_mask_pix.flatten(), pred_label_pix.flatten(), preds=pred_label_pix.flatten())
        print('pix', class_name, 'affiliation', aff_p_pix, aff_r_pix, aff_f1_pix)
        # pix_auc = precision_score(gt_mask_pix.flatten(), pred_label_pix.flatten())
        # pix_rec = recall_score(gt_mask_pix.flatten(), pred_label_pix.flatten())
        # pix_f1 = f1_score(gt_mask_pix.flatten(), pred_label_pix.flatten())
        # print('pix', class_name, 'auc', pix_auc, 'rec', pix_rec, 'f1', pix_f1)

        # ======img evaluation======
        pred_label_img = np.max(pred_label, axis=(1, 2))
        gt_label = np.asarray(gt_label_list, dtype=bool)
        
        img_auc = precision_score(gt_label, pred_label_img)
        img_rec = recall_score(gt_label, pred_label_img)
        img_f1 = f1_score(gt_label, pred_label_img)
        print('img', class_name, 'auc', img_auc, 'rec', img_rec, 'f1', img_f1)
        
        img_scores = np.max(scores, axis=(1, 2))
        if vis:
            print('vis', class_name, 'ia', img_auc, 'pa', aff_p_pix)

            precision, recall, thresholds = precision_recall_curve(gt_label, img_scores)
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            
            vis_path = f'{self.args.root_path}/vis_results/{self.args.save_prefix}/{class_name}'
            visulizer = Visualizer(vis_path)
            max_score = np.max(scores)
            min_score = np.min(scores)
            scores = (scores - min_score) / (max_score - min_score)
            test_imgs = np.concatenate(test_imgs, axis=0)
            if decoder is not None:
                decode_list = np.concatenate(decode_list, axis=0)
            visulizer.plot(test_imgs, scores, gt_mask, file_names, img_anomalies, decode_list)

        return img_f1, aff_f1_pix
    
    def save_pred_label_as_gray(self, pred_label, gt_mask, file_names, save_path):
        os.makedirs(save_path, exist_ok=True)
        for i, label in enumerate(pred_label):
            if gt_mask[i].max() == 0:
                continue
            # 归一化并转化为灰度图格式
            label = (label * 255).astype(np.uint8)
            # 保存图像
            file_name = os.path.basename(file_names[i])
            save_file = os.path.join(save_path, f"{os.path.splitext(file_name)[0]}_pred_label.png")
            cv2.imwrite(save_file, label)
        print(f"Saved {len(pred_label)} gray images to {save_path}")
    
    def filter_pred_label(self, pred_label, threshold_ratio=1/3):
        """
        过滤 pred_label 数据，基于像素点中 1 的比例判断。
        
        只有当某个像素点中，1 的比例大于等于 threshold_ratio 时，结果才设置为 1。
        
        参数：
        - pred_label: np.ndarray，形状为 (N, H, W)，每个像素的预测标签（0 或 1）
        - threshold_ratio: float，控制判定为 1 的最小比例
        
        返回：
        - filtered_label: np.ndarray，形状为 (N, H, W)，筛选后的结果
        """
        # 计算每个像素位置中值为 1 的比例
        count_ones = np.sum(pred_label == 1, axis=-1)  # 统计 1 的数量
        total = pred_label.shape[-1]  # 计算总数
        proportion = count_ones / total  # 计算比例

        # 判断是否超过阈值
        filtered_label = (proportion >= threshold_ratio).astype(np.uint8)
        return filtered_label

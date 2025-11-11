import cv2
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter

from loss import FocalLoss, BinaryDiceLoss
from tools import visualization, calculate_metric, calculate_average_metric
from .mvps import *
from .custom_clip import create_model_and_transforms
from tqdm import tqdm

from torchvision.utils import save_image
from torchvision.transforms.functional import resize
import os
from matplotlib import pyplot as plt
import cv2
import random

def prompt_regularization_loss_contrastive(
    prompt_normal, 
    prompt_abnormal,
    patch_tokens, 
    gt_mask, 
    is_anomaly,
    token_id, 
    eps=1e-8,
    lambda_normal=1.0, 
    lambda_abnormal=1.0, 
    lambda_orth=1.0
):
    B, N, C = patch_tokens.shape
    device = patch_tokens.device
    
    if len(gt_mask.shape) == 2:
        gt_mask = gt_mask.unsqueeze(0)

    if N == 1370:  # 去掉 cls token
        patch_tokens = patch_tokens[:,1:,:]
        token_size = int((N-1) ** 0.5)
    else:
        token_size = int(N ** 0.5)

    patch_tokens = patch_tokens.view(B, token_size, token_size, C)

    loss_total = 0.0
    for b in range(B):
        mask_resized = F.interpolate(
            gt_mask[b].float().unsqueeze(0).unsqueeze(0),
            size=(token_size, token_size),
            mode='nearest'
        ).squeeze()

        normal_tokens   = patch_tokens[b][mask_resized <= 0.5].reshape(-1, C)
        abnormal_tokens = patch_tokens[b][mask_resized > 0.5].reshape(-1, C)

        if is_anomaly[b] == 1:
            # 异常样本
            # abnormal proto ↔ abnormal tokens
            if abnormal_tokens.numel() > 0:  # 避免无 normal 区域
                sim_abn = torch.matmul(
                    F.normalize(abnormal_tokens, dim=-1),
                    F.normalize(prompt_abnormal[b], dim=-1).T
                )
                loss_total += lambda_abnormal * (1 - sim_abn.max(dim=1)[0].mean())

            # normal proto ↔ normal tokens
            sim_norm = torch.matmul(
                F.normalize(normal_tokens, dim=-1),
                F.normalize(prompt_normal[b], dim=-1).T,
            )
            loss_total += lambda_normal * (1 - sim_norm.max(dim=1)[0].mean())
            # print("ABNORMAL : ", lambda_abnormal * (1 - sim_abn.max(dim=1)[0].mean()), lambda_normal * (1 - sim_norm.max(dim=1)[0].mean()))
        else:
            # 正常样本
            # abnormal proto → 压缩到 0
            # loss_total += lambda_zero * torch.norm(prompt_abnormal[b], p=2) / (
            #     torch.sqrt(torch.tensor(C, device=device)) + eps
            # )
            # abnormal proto 与 normal tokens 正交
            sim_orth = torch.matmul(
                F.normalize(normal_tokens, dim=-1),
                F.normalize(prompt_abnormal[b], dim=-1).T
            )
            loss_total += lambda_orth * (sim_orth ** 2).mean()

            # normal proto ↔ 全部 tokens (因为全是 normal 区域)
            sim_norm = torch.matmul(
                F.normalize(normal_tokens, dim=-1),
                F.normalize(prompt_normal[b], dim=-1).T
            )
            loss_total += lambda_normal * (1 - sim_norm.max(dim=1)[0].mean())
            # print("NORMAL : ", lambda_orth * (sim_orth ** 2).mean(), lambda_normal * (1 - sim_norm.max(dim=1)[0].mean()))
    return loss_total

def save_combined_colormap(sim_maps, save_path, normalize=True, colormap=cv2.COLORMAP_JET):
    heatmaps = []
    for sim_map in sim_maps:
        sim_map = sim_map.astype(np.float32)
        if normalize:
            sim_map -= sim_map.min()
            sim_map /= (sim_map.max() + 1e-6)
        sim_map = (sim_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(sim_map, colormap)
        heatmaps.append(heatmap)
    
    combined = np.concatenate(heatmaps, axis=1)  # 横向拼接
    cv2.imwrite(save_path, combined)

class MVPS_Trainer(nn.Module):
    def __init__(
            self,
            # clip-related
            backbone, feat_list, input_dim, output_dim,
            # learning-related
            learning_rate, device, image_size,
            # model settings
            prompting_depth=5, 
            prompting_length=5,
            prompting_branch='VL', 
            prompting_type='SD',
            use_hsf=True, 
            alpha=.5,
            k_clusters=0.1
    ):

        super(MVPS_Trainer, self).__init__()

        self.device = device
        self.feat_list = feat_list
        self.image_size = image_size
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.k_clusters = k_clusters

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        ########### different model choices
        freeze_clip, _, self.preprocess = create_model_and_transforms(backbone, image_size,
                                                                      pretrained='openai')
        freeze_clip  = freeze_clip.half().to(device)
        freeze_clip.eval()
        
        self.clip_model = MVPS(freeze_clip=freeze_clip,
                                text_channel=output_dim,
                                visual_channel=input_dim,
                                prompting_length=prompting_length,
                                prompting_depth=prompting_depth,
                                prompting_branch=prompting_branch,
                                prompting_type=prompting_type,
                                use_hsf=use_hsf,
                                k_clusters=k_clusters,
                                output_layers=feat_list,
                                device=device,
                                image_size=image_size,
                                alpha=alpha).to(device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.preprocess.transforms[0] = transforms.Resize(size=(image_size, image_size),
                                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                                          max_size=None)

        self.preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))

        # update parameters
        self.learnable_paramter_list = [
            'text_prompter',
            'visual_prompter',
            'patch_token_layer',
            'cls_token_layer',
            'dynamic_visual_prompt_generator',
            'dynamic_text_prompt_generator',
            'hmpt',
            'visual_scale_prompt',
            # 'fusion_decoder',
            # 'hmpt_global',
            # 'hmpt_finegrain',
            'DPTO',
        ]

        self.params_to_update = []
        for name, param in self.clip_model.named_parameters():
            # print(name)
            for update_name in self.learnable_paramter_list:
                if update_name in name:
                    # print(f'updated parameters--{name}: {update_name}')
                    self.params_to_update.append(param)

        # build the optimizer
        self.optimizer = torch.optim.AdamW(self.params_to_update, lr=learning_rate, betas=(0.5, 0.999))
        self.epoch  = 0
        self.image_id = 0

    def save(self, path):
        self.save_dict = {}
        for param, value in self.state_dict().items():
            for update_name in self.learnable_paramter_list:
                if update_name in param:
                    # print(f'{param}: {update_name}')
                    self.save_dict[param] = value
                    break

        torch.save(self.save_dict, path)
    
    @torch.no_grad()
    def save_visualization(self, image, gt, qk_tokens_list, vv_tokens_list,
                    qk_attn_list, vv_attn_list, save_dir, image_id='000'):
        os.makedirs(save_dir, exist_ok=True)
        B, C, H, W = image.shape
        if B == 1: return
        if len(qk_attn_list) == 0 or len(vv_attn_list) == 0: return
        
        def get_patch_center_token(gt_mask, patch_h, patch_w):
            # 获取所有异常点的位置
            if len(gt_mask.shape) == 3: gt_mask = gt_mask[0]
            y, x = torch.nonzero(gt_mask, as_tuple=True)
            
            if len(y) == 0:
                # 没有异常点，返回 None 或者一个默认值
                return 0

            # 随机选取一个异常点
            idx = random.randint(0, len(y) - 1)
            center_y, center_x = int(y[idx].item()), int(x[idx].item())
            # 映射到 patch grid 上
            token_y = center_y * patch_h // gt_mask.shape[-2]
            token_x = center_x * patch_w // gt_mask.shape[-1]
            token_idx = int(token_y * patch_w + token_x)

            return token_idx
        
        # 获取 GT 中心的 token index
        num_patch = qk_attn_list[0].shape[1]
        patch_size = int((H * W / num_patch)**0.5)  # assume square
        grid_size = int(num_patch ** 0.5)
        center_token_idx = get_patch_center_token(gt[0], grid_size, grid_size)
        
        # for stage_idx, (qk_token, vv_token, qk_attn, vv_attn) in enumerate(zip(qk_tokens_list, vv_tokens_list, qk_attn_list, vv_attn_list)):
        for stage_idx, (qk_attn, vv_attn) in enumerate(zip(qk_attn_list, vv_attn_list)):
            # print(qk_attn.shape, vv_attn.shape)

            # token shape: [1375, 4, 1024] -> [visual_token=1369, head=4, dim]
            # qk_token = qk_token[1:1370].detach()  # remove global and learnable tokens
            # vv_token = vv_token[1:1370].detach()
            qk_attn = qk_attn[:, 1:1370, 1:1370].detach()  # remove attention to global/learnable
            vv_attn = vv_attn[:, 1:1370, 1:1370].detach()

            # 抽取出 attn map 的行
            center_vec_qk = qk_attn[0, center_token_idx,:].unsqueeze(0)
            center_vec_vv = vv_attn[0, center_token_idx,:].unsqueeze(0)
            center_vec_qk = center_vec_qk.view(grid_size, grid_size).cpu().numpy()
            center_vec_vv = center_vec_vv.view(grid_size, grid_size).cpu().numpy()

            sim_maps = [center_vec_qk, center_vec_vv]
            save_path = os.path.join(save_dir, f'{image_id}_stage{stage_idx}_attn0_combined.png')
            save_combined_colormap(sim_maps, save_path)

        # for b in range(image.shape[0]):
        # 保存第 b 个样本的输入图像
        save_image(image[0], os.path.join(save_dir, f'{image_id}_input.png'))
        # 保存 GT（ground truth）图像，确保为 numpy 格式
        gt_np = gt[0].cpu().numpy()
        if gt_np.ndim == 3:  # 若为 [1, H, W]，取出通道维
            gt_np = gt_np[0]
        # plt.imsave(os.path.join(save_dir, f'{image_id}_b{b}_gt.png'), gt_np, cmap='gray')
        gt_np = (gt_np * 255).astype(np.uint8)  # 若为 0~1 范围
        save_path = os.path.join(save_dir, f'{image_id}_gt.png')
        cv2.imwrite(save_path, gt_np)
        self.clip_model.empty_visualize_cache()

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def train_one_batch(self, items):
        image = items['img'].to(self.device)
        cls_name = items['cls_name']
        
        # losses
        gt = items['img_mask'].to(self.device)
        gt = gt.squeeze()

        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        is_anomaly = items['anomaly'].to(self.device)
        is_anomaly[is_anomaly > 0.5] = 1
        is_anomaly[is_anomaly <= 0.5] = 0

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(): # 开启混合精度训练
            # pixel level
            anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=False)

            if not isinstance(anomaly_map, list):
                anomaly_map = [anomaly_map]

            loss = 0

            # classification loss
            if isinstance(anomaly_score, list):
                classification_loss = 0
                for anomaly_score_ in anomaly_score[len(self.feat_list):]:
                    classification_loss += self.loss_focal(anomaly_score_, is_anomaly.unsqueeze(1))
                loss += 0.2 * classification_loss
                classification_loss = 0
                for anomaly_score_ in anomaly_score[:len(self.feat_list)]:
                    classification_loss += self.loss_focal(anomaly_score_, is_anomaly.unsqueeze(1))
                loss += 0.8 * classification_loss
            else:
                classification_loss = self.loss_focal(anomaly_score, is_anomaly.unsqueeze(1))
                loss += classification_loss           

            # seg loss
            seg_loss = 0
            for am , in zip(anomaly_map):
                if gt.shape == 2:
                    gt = gt.unsqeeze(0)
                seg_loss += (self.loss_focal(am, gt) + self.loss_dice(am[:, 1, :, :], gt) + self.loss_dice(am[:, 0, :, :], 1-gt))
            loss += seg_loss          
            
            # 新的损失函数去找到提示原型的
            # if self.clip_model.hmpt[0].prompt_normal is not None:
            #     for token_id in range(len(self.feat_list)):
            #         loss += 0.1 * \
            #             prompt_regularization_loss_contrastive(
            #                 prompt_normal  =self.clip_model.hmpt[token_id].prompt_normal,
            #                 prompt_abnormal=self.clip_model.hmpt[token_id].prompt_abnormal,
            #                 patch_tokens   =self.clip_model.hmpt[token_id].patch_token,  # 高层 patch tokens
            #                 gt_mask=gt,
            #                 is_anomaly=is_anomaly,
            #                 token_id=token_id
            #             )
            
            self.image_id += 1
            self.save_visualization(image, gt, self.clip_model.qk_token_visualize, self.clip_model.vv_tokne_visualize, self.clip_model.qk_attn_map_visualize, self.clip_model.vv_attn_map_visualize, save_dir="trainning_tmp_figure", image_id=f"{self.image_id:05}_{self.epoch:02}")
            self.clip_model.empty_visualize_cache()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, classification_loss, seg_loss

    def train_epoch(self, loader):
        self.clip_model.train()
        loss_list = []
        seg_loss_list = []
        classification_loss_list = []
        self.image_id = 0
        for items in tqdm(loader):
            loss = self.train_one_batch(items)
            loss_list.append(loss[0].item())
            classification_loss_list.append(loss[1].item())
            seg_loss_list.append(loss[2].item())

        print(f"SEG_LOSS : {np.mean(seg_loss_list)}, CLS_LOSS : {np.mean(classification_loss_list)}")
        return np.mean(loss_list)

    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None):
        self.clip_model.eval()

        results = {}
        results['cls_names'] = []
        results['imgs_gts'] = []
        results['anomaly_scores'] = []
        results['imgs_masks'] = []
        results['anomaly_maps'] = []
        results['imgs'] = []
        results['names'] = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_indx = 0
            for indx, items in enumerate(tqdm(dataloader)):
                if save_fig:
                    path = items['img_path']
                    for _path in path:
                        vis_image = cv2.resize(cv2.imread(_path), (self.image_size, self.image_size))
                        results['imgs'].append(vis_image)
                    cls_name = items['cls_name']
                    for _cls_name in cls_name:
                        image_indx += 1
                        results['names'].append('{:}-{:03d}'.format(_cls_name, image_indx))

                image = items['img'].to(self.device)
                cls_name = items['cls_name']
                results['cls_names'].extend(cls_name)
                gt_mask = items['img_mask']
                gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

                for _gt_mask in gt_mask:
                    results['imgs_masks'].append(_gt_mask.squeeze(0).numpy())  # px

                # pixel level
                anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=True)

                anomaly_map = anomaly_map.cpu().numpy()
                anomaly_score = anomaly_score.cpu().numpy()

                for _anomaly_map, _anomaly_score in zip(anomaly_map, anomaly_score):
                    _anomaly_map = gaussian_filter(_anomaly_map, sigma=4)
                    results['anomaly_maps'].append(_anomaly_map)
                    results['anomaly_scores'].append(_anomaly_score)

                is_anomaly = np.array(items['anomaly'])
                for _is_anomaly in is_anomaly:
                    results['imgs_gts'].append(_is_anomaly)

                self.image_id += 1
                self.save_visualization(image, gt_mask, self.clip_model.qk_token_visualize, self.clip_model.vv_tokne_visualize, self.clip_model.qk_attn_map_visualize, self.clip_model.vv_attn_map_visualize, save_dir="evaluation_tmp_figure", image_id=f"{self.image_id:05}")
                self.clip_model.empty_visualize_cache()



        # visualization
        if save_fig:
            print('saving fig.....')
            visualization.plot_sample_cv2(
                results['names'],
                results['imgs'],
                {'MVPS': results['anomaly_maps']},
                results['imgs_masks'],
                save_fig_dir
            )

            # ====== 新增 t-SNE 可视化 ======
            # print('saving t-SNE visualization.....')

            # for idx in range(len(self.clip_model.text_features)): 
            #     # 把 list 拼成一个整体矩阵
            #     all_features = np.concatenate(self.clip_model.text_features[idx], axis=0)  # (M, 768)
            #     all_labels = np.concatenate(self.clip_model.labels, axis=0)           # (M,)
            #     all_class_name = np.array(self.clip_model.class_name)                 # (M,)
                
            #     # 调用 visualize_feature
            #     visualization.visualize_feature(
            #         all_features,
            #         all_labels,
            #         legends=["Normal", "Anomaly"],   # 这里固定 normal/anomaly 两类
            #         n_components=2,
            #         method="TSNE",
            #         className = all_class_name
            #     )

            #     # 保存图片
            #     plt.savefig(os.path.join("./", f"tsne_text_features_{idx}.png"), dpi=300)
            #     plt.close()


        metric_dict = dict()
        for obj in obj_list:
            metric_dict[obj] = dict()

        for obj in obj_list:
            metric = calculate_metric(results, obj)
            obj_full_name = f'{obj}'
            metric_dict[obj_full_name] = metric

        metric_dict['Average'] = calculate_average_metric(metric_dict)

        return metric_dict
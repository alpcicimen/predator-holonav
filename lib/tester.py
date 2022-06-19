import copy

from lib.trainer import Trainer
import os, torch
from tqdm import tqdm
import numpy as np
from lib.benchmark_utils import ransac_pose_estimation, random_sample, get_angle_deviation, to_o3d_pcd, to_array
import open3d as o3d

# Modelnet part
from common.math_torch import se3
from common.math.so3 import dcm2euler
from common.misc import prepare_logger
from collections import defaultdict
import coloredlogs

import pyvista as pv

from scripts.demo import draw_registration_result

import pickle

class IndoorTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)
    
    def test(self):
        print('Start to evaluate on test datasets...')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}',exist_ok=True)

        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch
                inputs = c_loader_iter.next()
                ##################################
                # load inputs to device.
                for k, v in inputs.items():  
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)
                ###############################################
                # forward pass
                feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                pcd = inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                correspondence = inputs['correspondences']

                src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]

                data = dict()
                data['pcd'] = pcd.cpu()
                data['feats'] = feats.detach().cpu()
                data['overlaps'] = scores_overlap.detach().cpu()
                data['saliency'] = scores_saliency.detach().cpu()
                data['len_src'] = len_src
                data['rot'] = c_rot.cpu()
                data['trans'] = c_trans.cpu()

                torch.save(data,f'{self.snapshot_dir}/{self.config.benchmark}/{idx}.pth')



class KITTITester(Trainer):
    """
    KITTI tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)
    
    def test(self):
        print('Start to evaluate on test datasets...')
        tsfm_est = []
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        
        self.model.eval()
        rot_gt, trans_gt =[],[]
        with torch.no_grad():
            for _ in tqdm(range(num_iter)): # loop through this epoch
                inputs = c_loader_iter.next()
                ###############################################
                # forward pass
                for k, v in inputs.items():  
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)

                feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                scores_overlap = scores_overlap.detach().cpu()
                scores_saliency = scores_saliency.detach().cpu()

                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                rot_gt.append(c_rot.cpu().numpy())
                trans_gt.append(c_trans.cpu().numpy())
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]
                src_pcd , tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
                src_overlap, tgt_overlap = scores_overlap[:len_src], scores_overlap[len_src:]
                src_saliency, tgt_saliency = scores_saliency[:len_src], scores_saliency[len_src:]

                n_points = 5000
                ########################################
                # run random sampling or probabilistic sampling
                # src_pcd, src_feats = random_sample(src_pcd, src_feats, n_points)
                # tgt_pcd, tgt_feats = random_sample(tgt_pcd, tgt_feats, n_points)

                src_scores = src_overlap * src_saliency
                tgt_scores = tgt_overlap * tgt_saliency

                if(src_pcd.size(0) > n_points):
                    idx = np.arange(src_pcd.size(0))
                    probs = (src_scores / src_scores.sum()).numpy().flatten()
                    idx = np.random.choice(idx, size= n_points, replace=False, p=probs)
                    src_pcd, src_feats = src_pcd[idx], src_feats[idx]
                if(tgt_pcd.size(0) > n_points):
                    idx = np.arange(tgt_pcd.size(0))
                    probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                    idx = np.random.choice(idx, size= n_points, replace=False, p=probs)
                    tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

                ########################################
                # run ransac 
                distance_threshold = 0.3
                ts_est = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False, distance_threshold=distance_threshold, ransac_n = 4)
                tsfm_est.append(ts_est)
        
        tsfm_est = np.array(tsfm_est)
        rot_est = tsfm_est[:,:3,:3]
        trans_est = tsfm_est[:,:3,3]
        rot_gt = np.array(rot_gt)
        trans_gt = np.array(trans_gt)[:,:,0]

        rot_threshold = 5
        trans_threshold = 2

        np.savez(f'{self.snapshot_dir}/results',rot_est=rot_est, rot_gt=rot_gt, trans_est = trans_est, trans_gt = trans_gt)

        r_deviation = get_angle_deviation(rot_est, rot_gt)
        translation_errors = np.linalg.norm(trans_est-trans_gt,axis=-1)

        flag_1=r_deviation<rot_threshold
        flag_2=translation_errors<trans_threshold
        correct=(flag_1 & flag_2).sum()
        precision=correct/rot_gt.shape[0]

        message=f'\n Registration recall: {precision:.3f}\n'

        r_deviation = r_deviation[flag_1]
        translation_errors = translation_errors[flag_2]

        errors=dict()
        errors['rot_mean']=round(np.mean(r_deviation),3)
        errors['rot_median']=round(np.median(r_deviation),3)
        errors['trans_rmse'] = round(np.mean(translation_errors),3)
        errors['trans_rmedse']=round(np.median(translation_errors),3)
        errors['rot_std'] = round(np.std(r_deviation),3)
        errors['trans_std']= round(np.std(translation_errors),3)

        message+=str(errors)
        print(message)
        self.logger.write(message+'\n')



def compute_rigid_transform(a, b, weights):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    tsfm = torch.eye(4)
    tsfm[:3] = transform
    return tsfm.numpy()


def compute_metrics(data , pred_transforms):
    """
    Compute metrics required in the paper
    """
    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = data['transform_gt']
        points_src = data['points_src'][..., :3]
        points_ref = data['points_ref'][..., :3]
        points_raw = data['points_raw'][..., :3]

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        # # Modified Chamfer distance
        # src_transformed = se3.transform(pred_transforms, points_src)
        # ref_clean = points_raw
        # src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
        # dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        # dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        # chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        src_transformed = se3.transform(pred_transforms, points_src)

        dist_ref = torch.min(square_distance(points_ref, src_transformed), dim=-1)[0]

        chamfer_dist = torch.mean(dist_ref, dim=1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_array(t_mse),
            't_mae': to_array(t_mae),
            'err_r_deg': to_array(residual_rotdeg),
            'err_t': to_array(residual_transmag),
            'chamfer_dist': to_array(chamfer_dist)
        }

    return metrics

def print_metrics(logger, summary_metrics , losses_by_iteration=None,title='Metrics'):
    """Prints out formated metrics to logger"""

    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    logger.info('Chamfer error: {:.7f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']
    ))

def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized

def visualize(pred_transforms, data, index=0):

    ref_points = np.array((data['points_ref'])[index, :, :3])
    ref_normals = np.array((data['points_ref'])[index, :, 3:])
    source_points = (data['points_src'])[..., :3]
    source_normals = np.array((data['points_src'])[index, :, 3:])

    c_transforms = torch.from_numpy(pred_transforms).to('cpu')
    c_transform = c_transforms[index:(index+1), 4, :, :]

    src_transformed = np.array(se3.transform(c_transform, source_points))

    source_points = np.array(source_points)[index]

    pc = pv.PolyData(np.concatenate((source_points, ref_points)))
    pc['Normals'] = np.concatenate((source_normals, ref_normals))

    colors = np.concatenate((np.full(shape=len(ref_points), fill_value=1.0),
                             np.full(shape=len(source_points), fill_value=0.0)))

    pc['point_color'] = colors

    pc.plot(scalars='point_color')

    pc = pv.PolyData(np.concatenate((src_transformed[0], ref_points)))
    pc['Normals'] = np.concatenate((source_normals, ref_normals))

    # colors = np.concatenate((np.full(shape=len(raw_points), fill_value=1.0),
    #                          np.full(shape=len(source_points), fill_value=0.0)))

    pc['point_color'] = colors

    pc.plot(scalars='point_color')

def visualise(pred_transforms, data):
    # data = [data for data in tqdm(test_loader, leave=False)]

    # data = torch.from_numpy(test_loader.dataset._data).to('cpu')

    ref_points = np.array((data['points_ref'])[0, :, :3])
    ref_normals = np.array((data['points_ref'])[0, :, 3:])
    source_points = (data['points_src'])[..., :3]
    source_normals = np.array((data['points_src'])[0, :, 3:])

    # raw_points = np.array(data[0, :, :3])
    # raw_normals = np.array(data[0, :, 3:])
    # source_points = data[..., :3]
    # source_normals = np.array(data[0, :, 3:])

    # c_transforms = torch.from_numpy(pred_transforms).to('cpu')

    c_transform = pred_transforms[:, 0, :, :]

    src_transformed = np.array(se3.transform(c_transform, source_points))

    source_points = np.array(source_points)[0]

    pc0 = pv.PolyData(np.concatenate((source_points, ref_points)))
    pc0['Normals'] = np.concatenate((source_normals, ref_normals))

    colors0 = np.concatenate((np.full(shape=len(source_points), fill_value=1.0),
                             np.full(shape=len(ref_points), fill_value=0.0)))

    pc0['point_color'] = colors0

    pc0.plot(scalars='point_color')

    pc1 = pv.PolyData(np.concatenate((src_transformed[0], ref_points)))
    pc1['Normals'] = np.concatenate((source_normals, ref_normals))

    colors1 = np.concatenate((np.full(shape=len(source_points), fill_value=1.0),
                             np.full(shape=len(ref_points), fill_value=0.0)))

    pc1['point_color'] = colors1

    pc1.plot(scalars='point_color')

class ModelnetTester(Trainer):
    """
    Modelnet tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)   
    
    def test(self):
        print('Start to evaluate on test datasets...')
        _logger, _log_path = prepare_logger(self.config, log_path=os.path.join(self.snapshot_dir,'results'))

        pred_transforms = []
        total_rotation = []
        all_inlier_ratios = []

        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        stored_inputs = []
        
        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch
                inputs = c_loader_iter.next()
                stored_inputs.append(copy.deepcopy(inputs))
                try:
                    ##################################
                    # load inputs to device.
                    for k, v in inputs.items():  
                        if type(v) == list:
                            inputs[k] = [item.to(self.device) for item in v]
                        elif type(v) == dict:
                            pass
                        else:
                            inputs[k] = v.to(self.device)

                    rot_trace = inputs['sample']['transform_gt'][:, 0, 0] + inputs['sample']['transform_gt'][:, 1, 1] + \
                            inputs['sample']['transform_gt'][:, 2, 2]
                    rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
                    total_rotation.append(np.abs(to_array(rotdeg)))

                    ###################################
                    # forward pass
                    feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                    scores_overlap = scores_overlap.detach().cpu()
                    scores_saliency = scores_saliency.detach().cpu()

                    len_src = inputs['stack_lengths'][0][0]
                    src_feats, tgt_feats = feats[:len_src], feats[len_src:]
                    src_pcd , tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
                    src_overlap, tgt_overlap = scores_overlap[:len_src], scores_overlap[len_src:]
                    src_saliency, tgt_saliency = scores_saliency[:len_src], scores_saliency[len_src:]

                    
                    ########################################
                    # run probabilistic sampling
                    n_points = 450
                    src_scores = src_overlap * src_saliency
                    tgt_scores = tgt_overlap * tgt_saliency

                    if(src_pcd.size(0) > n_points):
                        idx = np.arange(src_pcd.size(0))
                        probs = (src_scores / src_scores.sum()).numpy().flatten()
                        idx = np.random.choice(idx, size= n_points, replace=False, p=probs)
                        src_pcd, src_feats = src_pcd[idx], src_feats[idx]
                    if(tgt_pcd.size(0) > n_points):
                        idx = np.arange(tgt_pcd.size(0))
                        probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                        idx = np.random.choice(idx, size= n_points, replace=False, p=probs)
                        tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

                    ########################################
                    # run ransac 
                    distance_threshold = 0.025
                    ts_est = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False, distance_threshold=distance_threshold, ransac_n = 3)
                    #
                    # draw_registration_result(src_pcd, tgt_pcd, src_overlap, tgt_overlap,
                    #                          src_saliency, tgt_saliency, ts_est)
                except: # sometimes we left over with too few points in the bottleneck and our k-nn graph breaks
                    ts_est = np.eye(4)
                pred_transforms.append(ts_est)


        total_rotation = np.concatenate(total_rotation, axis=0)
        _logger.info(('Rotation range in data: {}(avg), {}(max)'.format(np.mean(total_rotation), np.max(total_rotation))))

        pred_transforms = torch.from_numpy(np.array(pred_transforms)).float()[:,None,:,:]
        
        c_loader_iter = self.loader['test'].__iter__()
        num_processed, num_total = 0, len(pred_transforms)
        metrics_for_iter = [defaultdict(list) for _ in range(pred_transforms.shape[1])]

        gt_transforms = np.array([np.concatenate((pc_data['rot'].cpu().numpy(), pc_data['trans'].cpu().numpy()), axis=1)
                         for pc_data in self.loader['test']])
        pred_transforms_array = pred_transforms.cpu().detach().numpy()[:, 0, :3, :]

        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch
                inputs = c_loader_iter.next()
    
                batch_size = 1
                for i_iter in range(pred_transforms.shape[1]):
                    cur_pred_transforms = pred_transforms[num_processed:num_processed+batch_size, i_iter, :, :]
                    metrics = compute_metrics(inputs['sample'], cur_pred_transforms)
                    # visualise(pred_transforms, inputs['sample'])
                    for k in metrics:
                        metrics_for_iter[i_iter][k].append(metrics[k])
                num_processed += batch_size

        for i_iter in range(len(metrics_for_iter)):
            metrics_for_iter[i_iter] = {k: np.concatenate(metrics_for_iter[i_iter][k], axis=0)
                                        for k in metrics_for_iter[i_iter]}
            summary_metrics = summarize_metrics(metrics_for_iter[i_iter])
            print_metrics(_logger, summary_metrics, title='Evaluation result (iter {})'.format(i_iter))

        np.save(os.path.join(self.snapshot_dir, 'output/gt_transforms.npy'), gt_transforms)
        np.save(os.path.join(self.snapshot_dir, 'output/pred_transforms.npy'), pred_transforms_array)

        src_pcds = np.array([stored_input['sample']['points_src'].numpy() for stored_input in stored_inputs])
        tgt_pcds = np.array([stored_input['sample']['points_ref'].numpy() for stored_input in stored_inputs])

        np.save(os.path.join(self.snapshot_dir, 'output/src_pcds.npy'), src_pcds)
        np.save(os.path.join(self.snapshot_dir, 'output/tgt_pcds.npy'), tgt_pcds)


        # visualize(pred_transforms, inputs['sample'], index=3)

        

def get_trainer(config):
    if(config.dataset == 'indoor'):
        return IndoorTester(config)
    elif(config.dataset == 'kitti'):
        return KITTITester(config)
    elif(config.dataset == 'modelnet'):
        return ModelnetTester(config)
    elif(config.dataset == 'holonav'):
        return ModelnetTester(config)
    else:
        raise NotImplementedError

from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
import torch
if cfg.test.icp:
    from lib.utils.icp import icp_utils
    # from lib.utils.icp.icp_refiner.build import ext_
if cfg.test.un_pnp:
    from lib.csrc.uncertainty_pnp import un_pnp_utils
    import scipy
from PIL import Image
from lib.utils.img_utils import read_depth
from scipy import spatial
from lib.utils.vsd import inout
from transforms3d.quaternions import mat2quat, quat2mat
from lib.csrc.nn import nn_utils
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, jaccard_score

class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = os.path.join(result_dir, cfg.test.dataset)
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

        data_root = args['data_root']
        cls = cfg.cls_type
        model_path = os.path.join('data/linemod', cls, cls + '.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)
        self.diameter = linemod_config.diameters[cls] / 100

        self.proj2d = []
        self.add = []
        self.cmd5 = []

        self.icp_proj2d = []
        self.icp_add = []
        self.icp_cmd5 = []

        self.mask_ap = []
        self.pa_acc = []#我家的
        self.cohen = []#我家的
        self.jac_score = []#我家的
        self.f1 = []#我家的
        self.level_ad = []#我家的
        self.level_pr = []#我家的

        self.height = 480
        self.width = 640

        model = inout.load_ply(model_path)
        model['pts'] = model['pts'] * 1000
        self.icp_refiner = icp_utils.ICPRefiner(model, (self.width, self.height)) if cfg.test.icp else None
        # if cfg.test.icp:
        #     self.icp_refiner = ext_.Synthesizer(os.path.realpath(model_path))
        #     self.icp_refiner.setup(self.width, self.height)
        self.middle = [1,16,26,27,48,49,82,83,84,85,86,87,88,89,117,125,125,127,128,129,150,158,177,183,184,192,201,207,208,219,238,239,240,241,259,265,267,268,269,289,
                292,293,299,304,305,306,307,308,309,10,311,312,313,315,316,322,333,336,340,342,344,352,353,359,360,362,363,365,366,367,370,373,399,400,403,407,408,
                421,422,423,424,425,426,428,429,431,434,438,440,442,443,444,445,446,447,448,451,452,453,454,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,
                486,496,497,498,499,500,501,506,507,508,516,518,529,541,542,543,547,548,549,554,555,556,558,559,564,565,566,567,571,579,580,581,582,584,585,586,587,
                594,595,600,610,622,627,628,636,637,639,640,642,643,644,645,648,649,650,651,652,661,662,664,665,665,667,668,669,670,681,682,688,689,690,700,702,703,
                704,705,706,707,708,709,711,712,730,740,741,762,798,799,811,812,813,814,815,823,824,828,829,834,835,841,842,850,851,857,858,859,870,884,885,886,887,888,
                889,892,901,902,903,904,905,914,915,916,927,928,929,930,939,942,943,944,946,947,948,949,954,955,956,957,961,963,964,968,970,972,986,987,991,992,993,1004,
                1008,1009,1010,1011,1044,1045,1046,1047,1048,1049,1050,1051,1052,1055,1056,1060,1069,1070,1073,1082,1083,1084,1088,1089,1092,1093,1094,1095,1097,1100,1101,
                1111,1112,1113,1124,1127,1128,1129,1132,1134,1135,1154,1155,1156,1159,1164,1172,1175,1176,1177,1178,1179,1181,1182,1185]

        self.severe = [17,18,19,20,21,22,23,24,25,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,90,91,92,115,116,130,131,132,133,
                134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,199,200,209,210,211,212,213,214,215,216,217,218,220,221,222,223,224,225,226,
                227,228,229,230,231,232,233,234,242,243,262,263,264,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,288,291,295,296,317,318,
                319,320,321,323,324,325,326,327,328,329,330,331,332,335,337,338,339,341,343,345,346,347,348,349,350,351,354,355,361,364,368,369,371,
                372,409,410,411,412,413,427,455,456,457,458,459,479,480,481,482,483,484,485,487,488,489,490,502,503,504,505,509,510,511,512,513,514,515,
                544,545,546,550,551,552,553,557,560,561,562,563,568,569,570,590,591,592,593,596,597,598,609,611,612,613,614,615,616,617,618,619,620,623,
                629,630,632,633,638,641,646,647,653,654,655,656,660,663,671,672,673,674,675,676,677,679,680,698,699,701,710,722,723,723,724,725,726,727,
                728,729,733,734,735,736,737,738,739,759,760,761,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,
                802,803,804,808,810,816,817,818,819,820,821,822,825,826,827,830,831,832,833,843,846,847,848,849,852,860,861,871,872,873,874,875,876,877,878,879,880,
                881,891,895,896,897,898,899,900,906,907,908,909,910,911,912,913,917,918,919,920,921,922,923,924,925,926,934,940,941,945,950,951,952,953,958,959,960,961,
                965,966,967,969,971,973,974,975,976,977,978,979,1005,1006,1007,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,
                1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1053,1054,1074,1091,1102,1103,1104,1105,1106,1107,1108,1109,1110,1121,
                1122,1125,1126,1137,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1161,1162,1163,1165,1166,1167,1168,1169,1170,1171,1173,1174,1186,1187]

    def projection_2d(self, pose_pred, pose_targets, K, imgid = 0, icp=False, threshold=5):
        np.set_printoptions(threshold=np.inf)
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        #print("预测:",len(model_2d_pred))
        if icp:
            self.icp_proj2d.append(proj_mean_diff < threshold)
        else:
            self.proj2d.append(proj_mean_diff < threshold)
        # self.proj2d.append(proj_mean_diff)
        # fp1 = open('/home/ivclab/homework/(pvnet)2d_mean_error.txt', 'a')
        # fp1.write(str(proj_mean_diff) + '\n')
        # fp1.close()


    def add_metric(self, pose_pred, pose_targets, imgid=0, icp=False, syn=False, percentage=0.1):
        np.set_printoptions(threshold=np.inf)
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        #print("预测:",model_pred)
        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)
        # self.add.append(mean_dist)
        # self.proj2d.append(mean_dist)
        # fp = open('/home/ivclab/homework/(pvnet)add_mean_error.txt', 'a')
        # fp.write(str(mean_dist) + '\n')
        # fp.close()


    def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        if icp:
            self.icp_cmd5.append(translation_distance < 5 and angular_distance < 5)
        else:
            self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mask_iou(self, output, batch):

        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        #print("原始的", mask_pred.min(), mask_pred.max())

        #########################################################
        #print("mask_pred:",mask_pred.shape)
        # mask_pred[mask_pred > 0] = 0

        # mask_p = np.squeeze(mask_p)
        # # mask_p = np.expand_dims(mask_p,axis=2)
        # seg_p = np.squeeze(seg_p, axis=0)
        # seg_p = np.delete(seg_p, obj=1, axis=0)
        # seg_p = np.squeeze(seg_p, axis=0)
        # seg_p = np.expand_dims(seg_p, axis=2)
        # print(seg_p.shape, seg_p.min(), seg_p.max())
        # seg_p[seg_p > 0] = 0
        # print("mask原始",mask_p.shape)
        # plt.imshow(mask_gt)
        # plt.show()
        # plt.savefig('/home/ivclab/path/to/clean-pvnet/data/result/pvnet/fomycat/lm_gt/' + str(batch['img_id'][0]) + '.jpg')
        # pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu()
        # gt = batch['mask'][0].detach().cpu()
        # correct = torch.eq(pred, gt).int()
        # accuracy = float(correct.sum()) / float(correct.numel())
        # print("ac",accuracy)
        # plt.imshow(mask_pred)
        # plt.savefig('/home/ivclab/path/to/clean-pvnet/data/result/pvnet/mycat/lm_seg/' + str(batch['img_id'][0]) + '.jpg')
        #######################################################

        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        #
        self.mask_ap.append(iou > 0.7)
        # iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()

    def pixel_acc(self, output, batch):
        pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu()#[1, 2, 480, 640]
        gt = batch['mask'][0].detach().cpu()

        # pred_ver = torch.argmax(output['vertex'], dim=1)[0].detach().cpu()
        # gt_ver = batch['vertex'][0][0].detach().cpu()
        correct = torch.eq(pred, gt).int()


        accuracy = float(correct.sum()) / float(correct.numel())
        #accuracy = accuracy.numpy()
        # print("对比",pred_ver.size(), gt_ver.size())

        # plt.axis('off')
        # plt.margins(0, 0)
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.imshow(pred_ver)
        # plt.show()
        # plt.imshow(gt_ver)
        # plt.show()
        self.pa_acc.append(accuracy)
    def cohen_score(self, output, batch):
        pred = torch.argmax(output['seg'], dim=1)[0].detach()
        pred = pred.cpu().contiguous().view(-1).numpy()
        gt = batch['mask'][0].detach()
        ground_truth = gt.cpu().contiguous().view(-1).numpy()
        kapp_score = cohen_kappa_score(ground_truth, pred)
        self.cohen.append(kapp_score)
    def MIOU_value(self, output, batch):
        id = int(batch['img_id'][0])
        pred = torch.argmax(output['seg'], dim=1)[0].detach()
        pred = pred.cpu().contiguous().view(-1).numpy()
        gt = batch['mask'][0].detach()
        ground_truth = gt.cpu().contiguous().view(-1).numpy()
        MIOU = jaccard_score(ground_truth, pred, average='macro')
        # print("图片的id",id)
        # if id == 232 or id == 218 or id == 223 or id == 228:
        #     pass
        # else:
        #     self.jac_score.append(MIOU)
        self.jac_score.append(MIOU)
    def F1_score(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        Dice = 2 * (np.abs(mask_pred & mask_gt).sum()) / (np.abs(mask_pred).sum() + np.abs(mask_gt).sum())
        self.f1.append(Dice)

    def level_p2d(self, pose_pred, pose_targets, K, img_id, icp=False, threshold=5):
        # level = 1#重度
        # model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        # model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        # proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        # if icp:
        #     self.icp_proj2d.append(proj_mean_diff < threshold)
        # else:
        #     self.proj2d.append(proj_mean_diff < threshold)

        # if level == 1:
        #     for num in range(len(self.severe)):
        #         if img_id == self.severe[num]:
        #             proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        #             self.level_pr.append(proj_mean_diff < threshold)
        # else:
        #     pass
        # if img_id not in self.middle and img_id not in self.severe:
        #     proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        #     self.level_pr.append(proj_mean_diff < threshold)
        # else:
        #     pass
        np.set_printoptions(threshold=np.inf)
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        # print("预测:",model_2d_pred)
        if icp:
            self.icp_proj2d.append(proj_mean_diff < threshold)
        else:
            self.proj2d.append(proj_mean_diff < threshold)

        # fp = open('/home/ivclab/homework/2d_pred.txt', 'a')
        # ft = open('/home/ivclab/homework/2d_target.txt', 'a')
        # fp.writelines(str(img_id)+":")
        # fp.writelines(str(model_2d_pred))
        # ft.writelines(str(img_id)+":")
        # ft.writelines(str(model_2d_targets))
        # ft.close()
        # fp.close()


    def level_add(self, pose_pred, pose_targets, img_id, icp=False, syn=False, percentage=0.1):

        # level=1
        # diameter = self.diameter * percentage
        # model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        # model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]
        #
        # # if level == 1:
        # #     for num in range(len(self.severe)):
        # #         if img_id == self.severe[num]:
        # #             mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        # #             self.level_ad.append(mean_dist < diameter)
        # # else:
        # #     pass
        #
        # if img_id not in self.middle and img_id not in self.severe:
        #     mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        #     self.level_ad.append(mean_dist < diameter)
        # else:
        #     pass

        np.set_printoptions(threshold=np.inf)
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        # print("预测:",model_pred)
        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

        # fp = open('/home/ivclab/homework/3d_pred.txt', 'a')
        # ft = open('/home/ivclab/homework/3d_target.txt', 'a')
        # fp.writelines(str(img_id) + ":")
        # fp.writelines(str(model_pred))
        # ft.writelines(str(img_id) + ":")
        # ft.writelines(str(model_targets))
        # ft.close()
        # fp.close()


    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

        R_refined, t_refined = self.icp_refiner.refine(depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(depth, R_refined, t_refined, K.copy(), no_depth=True)

        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))

        return pose_pred

    def uncertainty_pnp(self, kpt_3d, kpt_2d, var, K):
        cov_invs = []
        for vi in range(var.shape[0]):
            if var[vi, 0, 0] < 1e-6 or np.sum(np.isnan(var)[vi]) > 0:
                cov_invs.append(np.zeros([2, 2]).astype(np.float32))
            else:
                cov_inv = np.linalg.inv(scipy.linalg.sqrtm(var[vi]))
                cov_invs.append(cov_inv)

        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]
        pose_pred = un_pnp_utils.uncertainty_pnp(kpt_2d, weights, kpt_3d, K)

        return pose_pred

    def icp_refine_(self, pose, anno, output):
        depth = read_depth(anno['depth_path']).astype(np.uint16)
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask = mask.astype(np.int32)
        pose = pose.astype(np.float32)

        poses = np.zeros([1, 7], dtype=np.float32)
        poses[0, :4] = mat2quat(pose[:, :3])
        poses[0, 4:] = pose[:, 3]

        poses_new = np.zeros([1, 7], dtype=np.float32)
        poses_icp = np.zeros([1, 7], dtype=np.float32)

        fx = 572.41140
        fy = 573.57043
        px = 325.26110
        py = 242.04899
        zfar = 6.0
        znear = 0.25;
        factor= 1000.0
        error_threshold = 0.01

        rois = np.zeros([1, 6], dtype=np.float32)
        rois[:, :] = 1

        self.icp_refiner.solveICP(mask, depth,
            self.height, self.width,
            fx, fy, px, py,
            znear, zfar,
            factor,
            rois.shape[0], rois,
            poses, poses_new, poses_icp,
            error_threshold
        )

        pose_icp = np.zeros([3, 4], dtype=np.float32)
        pose_icp[:, :3] = quat2mat(poses_icp[0, :4])
        pose_icp[:, 3] = poses_icp[0, 4:]

        return pose_icp

    def evaluate(self, output, batch):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        if cfg.test.un_pnp:
            var = output['var'][0].detach().cpu().numpy()
            pose_pred = self.uncertainty_pnp(kpt_3d, kpt_2d, var, K)
        else:
            pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        if cfg.test.icp:
            pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K)
            if cfg.cls_type in ['eggbox', 'glue']:
                self.add_metric(pose_pred_icp, pose_gt, syn=True, icp=True)
            else:
                self.add_metric(pose_pred_icp, pose_gt, icp=True)
            self.projection_2d(pose_pred_icp, pose_gt, K, icp=True)
            self.cm_degree_5_metric(pose_pred_icp, pose_gt, icp=True)

        if cfg.cls_type in ['eggbox', 'glue']:
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt, img_id)

        self.projection_2d(pose_pred, pose_gt, K, img_id)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.mask_iou(output, batch)
        self.pixel_acc(output, batch)
        self.cohen_score(output, batch)
        self.MIOU_value(output, batch)
        self.F1_score(output, batch)
        # self.level_p2d(pose_pred, pose_gt, K,img_id)
        # self.level_add(pose_pred, pose_gt,img_id)


    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)
        pa = np.mean(self.pa_acc)
        ch = np.mean(self.cohen)
        ja = np.mean(self.jac_score)
        f1 = np.mean(self.f1)
        # le_pr = np.mean(self.level_pr)
        # le_ad = np.mean(self.level_ad)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('mask ap70: {}'.format(ap))
        print('mask pixel accuracy: {}'.format(pa))
        print('mask cohenscore: {}'.format(ch))
        print('mask MIOU: {}'.format(ja))
        # print('mask F1 score: {}'.format(f1))
        # print('level 2d score: {}'.format(le_pr))
        # print('level add score: {}'.format(le_ad))
        if cfg.test.icp:
            print('2d projections metric after icp: {}'.format(np.mean(self.icp_proj2d)))
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
            print('5 cm 5 degree metric after icp: {}'.format(np.mean(self.icp_cmd5)))
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.mask_ap = []
        self.pa_acc = []#我家的
        self.cohen = []#我家的
        self.jac_score = []  # 我家的
        self.f1 = []  # 我家的
        # self.level_ad= []#我家的
        # self.level_pr = []#我家的
        self.icp_proj2d = []
        self.icp_add = []
        self.icp_cmd5 = []
        return {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'ap': ap, 'pa_acc':pa, 'kapa_score':ch, 'MIOU':ja, 'F1_score':f1}#, 'level_pr':le_pr, 'level_ad':le_ad}

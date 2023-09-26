from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        #####
        # seg_p = output['seg'].cpu().numpy()
        # mask_p = output['mask'].cpu().numpy()
        # print("输出的总大小：",output.keys())#['seg', 'vertex', 'mask', 'kpt_2d']
        # plt.imshow(batch['mask'].cpu().numpy())

        # plt.subplot(222)
        # mask_p = np.squeeze(mask_p)
        # mask_p = np.expand_dims(mask_p,axis=2)
        # plt.imshow(mask_p)
        # plt.show()

        # seg_p = np.squeeze(seg_p,axis=0)
        # seg_p = np.delete(seg_p,obj=1,axis=0)
        # seg_p = np.squeeze(seg_p, axis=0)
        # seg_p = np.expand_dims(seg_p, axis=2)
        # print(seg_p.shape, seg_p.min(), seg_p.max())
        # seg_p[seg_p > 0] = 0
        # plt.imshow(-seg_p)#jet, hsv, pink
        # #plt.imshow(-seg_p, cmap='gray')
        # plt.show()
        # plt.savefig('/home/ivclab/path/to/clean-pvnet/data/result/pvnet/mycat/segmentation/' + str(batch['img_id'][0]) + '.jpg')
        # print("seg形状", mask_p.shape)
        #####
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        # plt.savefig('/home/ivclab/path/to/clean-pvnet/data/result/pvnet/fomyape/LinemodOccTest/' + str(img_id) + '.jpg')
        plt.show()

    def visualize_demo(self, output, inp, meta):
        inp = img_utils.unnormalize_img(inp[0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        kpt_3d = np.array(meta['kpt_3d'])
        K = np.array(meta['K'])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(meta['corner_3d'])
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()

    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        #plt.savefig('test.jpg')
        plt.close(0)






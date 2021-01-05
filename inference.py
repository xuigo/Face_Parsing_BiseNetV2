"""
Test bisenetv2 on celebamaskhq dataset
"""
import os
import sys
import tqdm
import cv2
import yaml
import numpy as np
import tensorflow as tf
from bisenet_model import bisenet_v2

CFG = yaml.load(open('config.yaml'))


SRC_DIR = './test'
DST_DIR = './res'
SELECT_PARTS = [ 'skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'left_ear', 'right_ear','nose', 'mouth', 'up_lip', 'low_lip', 'hair','eye_glass']


class BiSeNetV2:

    def __init__(self):
        self.category_id = {'background': 0, 'skin': 1, 'left_brow': 2, 'right_brow': 3, 'left_eye': 4, 'right_eye': 5, 'eye_glass': 6, 'left_ear': 7,
                            'right_ear': 8, 'earring': 9, 'nose': 10, 'mouth': 11, 'up_lip': 12, 'low_lip': 13, 'neck': 14, 'necklace': 15, 'cloth': 16, 'hair': 17, 'hat': 18}

        self.label_contours = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                               (128, 0, 128), (0, 128, 128), (128,
                                                              128, 128), (64, 0, 0), (192, 0, 0),
                               (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0)]
        #self.model_path = weight
        self.select = SELECT_PARTS
        self.model_path='./checkpoints/face'
        self.select_id = []
        self._select_check()
        self.size = [int(tmp) for tmp in CFG['AUG']['EVAL_CROP_SIZE']]
        self.build_model()

    def _select_check(self):
        for key in self.select:
            if not self.category_id.get(key, None):
                raise ValueError("%s doesn't exist in Category List! Which contains:\n %s" % (
                    key, [key for key in self.category_id]))
            else:
                self.select_id.append(self.category_id.get(key))

    def _dir_check(self, src, dst):
        if not os.path.isdir(src):
            raise ValueError("%s is not a correct Path!" % src)
        os.makedirs(dst, exist_ok=True)

    def backrpocess(self, mask):
        mask_shape = mask.shape
        mask_color = np.zeros(
            shape=[mask_shape[0], mask_shape[1], 3], dtype=np.uint8)
        unique_label_ids = [v for v in np.unique(mask) if v != 0 and v != 255]
        for label_id in unique_label_ids:
            if label_id in self.select_id:
                idx = np.where(mask == label_id)
                #mask_color[idx] = self.label_contours[label_id]
                mask_color[idx] = self.recovery_image[:, :, (2, 1, 0)][idx]
        return mask_color

    def preprocess(self, src_image):
        output_image = src_image[:, :, (2, 1, 0)]
        self.recovery_image = cv2.resize(output_image, dsize=(
            self.size[0], self.size[1]), interpolation=cv2.INTER_LINEAR)
        output_image = self.recovery_image.astype('float32') / 255.0
        img_mean = np.array(CFG['DATASET']['MEAN_VALUE']).reshape(
            (1, 1, len(CFG['DATASET']['MEAN_VALUE'])))
        img_std = np.array(CFG['DATASET']['STD_VALUE']).reshape(
            (1, 1, len(CFG['DATASET']['STD_VALUE'])))
        output_image -= img_mean
        output_image /= img_std
        return output_image
         
    def build_model(self):
        # define graph
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[
                                      1, self.size[1], self.size[0], 3], name='input_tensor')
        self.bisenet_model = bisenet_v2.BiseNetV2(phase='test', cfg=CFG)
        self.prediction = self.bisenet_model.inference(
            input_tensor=self.input_tensor, name='BiseNetV2', reuse=False)
        # define session and gpu config
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG['GPU']['GPU_MEMORY_FRACTION']
        sess_config.gpu_options.allow_growth = CFG['GPU']['TF_ALLOW_GROWTH']
        sess_config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=sess_config)
        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG['SOLVER']['MOVING_AVE_DECAY'])
            variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        saver.restore(self.sess, self.model_path)
        print(' [*] Model Restored.')

    @staticmethod
    def _get_files(src):
        if os.path.isdir(src):
            return [os.path.join(src,file) for file in os.listdir(src)]
        elif os.path.isfile(src):
            return [src]
        else:
            raise ValueError(' [!] Unavailable Path %s'%src)

    def test(self,src,dst):
        files=BiSeNetV2._get_files(src)
        for file in tqdm.tqdm(files):
            filepath, filename = os.path.split(file)
            src_image = cv2.imread(file, cv2.IMREAD_COLOR)
            preprocessed_image = self.preprocess(src_image)
            prediction_value = self.sess.run(fetches=self.prediction, feed_dict={
                                        self.input_tensor: [preprocessed_image]})
            prediction_value = np.squeeze(prediction_value, axis=0)
            prediction_mask_color = self.backrpocess(prediction_value)
            cv2.imwrite(os.path.join(dst, filename), prediction_mask_color)

if __name__ == '__main__':
    bisenet = BiSeNetV2()
    bisenet.test(SRC_DIR, DST_DIR)

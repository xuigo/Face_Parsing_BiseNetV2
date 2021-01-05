import os
import os.path as ops
import yaml
import tqdm
import random 
import cv2
import numpy as np

CFG = yaml.load(open('config.yaml'))

class data_reader:

    def __init__(self,data_dir,batch_size):

        self.labels = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
          'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        self.data_dir=data_dir
        self._check()
        self.img_num=30000
        #self.epoch=epoch
        self.batch_size=batch_size
        self.src_size=(448,448)
        self.label_size=(512,512)
        self.data_index=self._get_index()
        
    def _check(self):
        self.source_image_dir=os.path.join(self.data_dir,'CelebA-HQ-img')
        self.anno_image_dir=os.path.join(self.data_dir,'CelebAMask-HQ-mask-anno')
        self.mask_manual=os.path.join(self.data_dir,'CelebAMask-Manual')
        assert os.path.exists(self.source_image_dir),"%s doesn't exists!"%self.source_image_dir
        assert os.path.exists(self.anno_image_dir),"%s doesn't exists!"%self.anno_image_dir
        os.makedirs(self.mask_manual,exist_ok=True)

    def _get_index(self):
        label_info = []
        for k in tqdm.tqdm(range(self.img_num)):
            #for k in tqdm.tqdm(range(2010)):
            output_label_image_save_path = os.path.join(self.mask_manual, '{:d}.png'.format(k))
            source_image_path = ops.join(self.source_image_dir, '{:d}.jpg'.format(k))
            assert os.path.exists(source_image_path), '{:s} not exist'.format(source_image_path)
            if ops.exists(output_label_image_save_path):
                label_info.append([source_image_path, output_label_image_save_path])
                continue
            
            folder_num = k // 2000
            im_base = np.zeros((512, 512))            
            gt=np.zeros((512,512,3),np.uint8)
            for idx, label in enumerate(self.labels):
                anno_file_path = ops.join(
                    self.anno_image_dir,
                    str(folder_num),
                    str(k).rjust(5, '0') + '_' + label + '.png'
                )
                if os.path.exists(anno_file_path):
                    im = cv2.imread(anno_file_path, cv2.IMREAD_UNCHANGED)
                    im = im[:, :, 0]               
                    im_base[im != 0] = (idx + 1)                            
                    gt[:,:,0]=im_base    
                    gt[:,:,1]=im_base 
                    gt[:,:,2]=im_base       
            cv2.imwrite(output_label_image_save_path, gt)
            label_info.append([source_image_path, output_label_image_save_path])
        
        return label_info
        
    def __len__(self):
        return len(self.data_index) // self.batch_size 

    def get_iter(self):
        #for i in range(self.epoch):
        random.shuffle(self.data_index)
        max_iter=(len(self.data_index)//self.batch_size) * self.batch_size
        
        for ii in range(0,max_iter,self.batch_size):
            srcs,labels=[],[]
            for jj in range(ii,(ii+1)*self.batch_size):
                src=cv2.resize(cv2.imread(self.data_index[jj][0]),self.src_size)
                src = src/255.
                src -= CFG['DATASET']['MEAN_VALUE']
                src /= CFG['DATASET']['STD_VALUE']
                label=cv2.resize(cv2.imread(self.data_index[jj][1]),self.label_size)
                srcs.append(src)
                labels.append(label)
            yield np.array(srcs),np.array(labels)

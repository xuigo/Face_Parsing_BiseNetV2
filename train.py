import os
import os.path as ops
import shutil
import time
import math
import yaml
import sys
import numpy as np
import tensorflow as tf
import loguru
import tqdm

from bisenet_model import bisenet_v2
from utils.augment import preprocess_image_for_train
from utils.data_reader import data_reader

LOG = loguru.logger

CFG = yaml.load(open('config.yaml'))

class BiseNetV2CelebamaskhqTrainer(object):
    """
    init bisenetv2 single gpu trainner
    """
    def __init__(self):
        # define solver params and dataset
        self.dataloader=data_reader(data_dir=CFG['DATASET']['DATA_DIR'],batch_size=CFG['TRAIN']['BATCH_SIZE'])
        self._steps_per_epoch = len(self.dataloader)
        self._model_name = CFG['MODEL']['MODEL_NAME']
        self._train_epoch_nums = CFG['TRAIN']['EPOCH_NUMS']
        self._batch_size = CFG['TRAIN']['BATCH_SIZE']
        self._snapshot_epoch = CFG['TRAIN']['SNAPSHOT_EPOCH']
        self._model_save_dir = ops.join(CFG['TRAIN']['MODEL_SAVE_DIR'], self._model_name)
        self._tboard_save_dir = ops.join(CFG['TRAIN']['TBOARD_SAVE_DIR'], self._model_name)
        self._enable_miou = CFG['TRAIN']['COMPUTE_MIOU']['ENABLE']
        if self._enable_miou:
            self._record_miou_epoch = CFG['TRAIN']['COMPUTE_MIOU']['EPOCH']
        self._input_tensor_size = [int(tmp) for tmp in CFG['AUG']['TRAIN_CROP_SIZE']]
        self._init_learning_rate = CFG['SOLVER']['LR']
        self._moving_ave_decay = CFG['SOLVER']['MOVING_AVE_DECAY']
        self._momentum = CFG['SOLVER']['MOMENTUM']
        self._lr_polynimal_decay_power = CFG['SOLVER']['LR_POLYNOMIAL_POWER']
        self._optimizer_mode = CFG['SOLVER']['OPTIMIZER'].lower()
        if CFG['TRAIN']['RESTORE_FROM_SNAPSHOT']['ENABLE']:
            self._initial_weight = CFG['TRAIN']['RESTORE_FROM_SNAPSHOT']['SNAPSHOT_PATH']
        else:
            self._initial_weight = None
        if CFG['TRAIN']['WARM_UP']['ENABLE']:
            self._warmup_epoches = CFG['TRAIN']['WARM_UP']['EPOCH_NUMS']
            self._warmup_init_learning_rate = self._init_learning_rate / 1000.0
        else:
            self._warmup_epoches = 0
        # define tensorflow session
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG['GPU']['GPU_MEMORY_FRACTION']
        sess_config.gpu_options.allow_growth = CFG['GPU']['TF_ALLOW_GROWTH']
        sess_config.gpu_options.allocator_type = 'BFC'
        self._sess = tf.Session(config=sess_config)

        self.src_input=tf.placeholder(dtype=tf.float32,shape=[None,448,448,3],name='src')
        self.label_input=tf.placeholder(dtype=tf.int32,shape=[None,512,512,1],name='label')
        
        # define model loss
        self._model = bisenet_v2.BiseNetV2(phase='train', cfg=CFG)
        
        loss_set = self._model.compute_loss(input_tensor=self.src_input, label_tensor=self.label_input, name='BiseNetV2',reuse=False)
        self._prediciton = self._model.inference(input_tensor=self.src_input,name='BiseNetV2',reuse=True)
        self._loss = loss_set['total_loss']
        self._l2_loss = loss_set['l2_loss']

        # define miou
        if self._enable_miou:
            with tf.variable_scope('miou'):
                pred = tf.reshape(self._prediciton, [-1, ])
                gt = tf.reshape(self.label_input, [-1, ])
                indices = tf.squeeze(tf.where(tf.less_equal(gt, CFG['DATASET']['NUM_CLASSES'] - 1)), 1)
                gt = tf.gather(gt, indices)
                pred = tf.gather(pred, indices)
                self._miou, self._miou_update_op = tf.metrics.mean_iou(labels=gt,predictions=pred,num_classes=CFG['DATASET']['NUM_CLASSES'])

        # define learning rate
        with tf.variable_scope('learning_rate'):
            self._global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
            warmup_steps = tf.constant(self._warmup_epoches * self._steps_per_epoch, dtype=tf.float32, name='warmup_steps')
            train_steps = tf.constant(self._train_epoch_nums * self._steps_per_epoch, dtype=tf.float32, name='train_steps')  
            self._learn_rate = tf.cond(
                pred=self._global_step < warmup_steps,
                true_fn=lambda: self._compute_warmup_lr(warmup_steps=warmup_steps, name='warmup_lr'),
                false_fn=lambda: tf.train.polynomial_decay(
                    learning_rate=self._init_learning_rate,
                    global_step=self._global_step,
                    decay_steps=train_steps,
                    end_learning_rate=0.000001,
                    power=self._lr_polynimal_decay_power)
            )
            self._learn_rate = tf.identity(self._learn_rate, 'lr')
            global_step_update = tf.assign_add(self._global_step, 1.0)

        # define moving average op
        with tf.variable_scope(name_or_scope='moving_avg'):
            if CFG['TRAIN']['FREEZE_BN']['ENABLE']:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            moving_ave_op = tf.train.ExponentialMovingAverage(
                self._moving_ave_decay).apply(train_var_list + tf.moving_average_variables())

        # define training op
        with tf.variable_scope(name_or_scope='train_step'):
            if CFG['TRAIN']['FREEZE_BN']['ENABLE']:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            if self._optimizer_mode == 'sgd':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self._learn_rate,
                    momentum=self._momentum
                )
            elif self._optimizer_mode == 'adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self._learn_rate,
                )
            else:
                raise ValueError('Not support optimizer: {:s}'.format(self._optimizer_mode))
            optimize_op = optimizer.minimize(self._loss, var_list=train_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([optimize_op, global_step_update]):
                    with tf.control_dependencies([moving_ave_op]):
                        self._train_op = tf.no_op()

        # define saver and loader
        with tf.variable_scope('loader_and_saver'):
            self._net_var = [vv for vv in tf.global_variables() if 'lr' not in vv.name]
            self._loader = tf.train.Saver(self._net_var)
            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # define summary
        with tf.variable_scope('summary'):
            summary_merge_list = [
                tf.summary.scalar("learn_rate", self._learn_rate),
                tf.summary.scalar("total", self._loss),
                tf.summary.scalar('l2_loss', self._l2_loss)
            ]
            if self._enable_miou:
                with tf.control_dependencies([self._miou_update_op]):
                    summary_merge_list_with_miou = [
                        tf.summary.scalar("learn_rate", self._learn_rate),
                        tf.summary.scalar("total", self._loss),
                        tf.summary.scalar('l2_loss', self._l2_loss),
                        tf.summary.scalar('miou', self._miou)
                    ]
                    self._write_summary_op_with_miou = tf.summary.merge(summary_merge_list_with_miou)
            if ops.exists(self._tboard_save_dir):
                shutil.rmtree(self._tboard_save_dir)
            os.makedirs(self._tboard_save_dir, exist_ok=True)
            # model_params_file_save_path = ops.join(self._tboard_save_dir, CFG['TRAIN']['MODEL_PARAMS_CONFIG_FILE_NAME'])
            # with open(model_params_file_save_path, 'w', encoding='utf-8') as f_obj:
            #     CFG['dump_to_json_file(f_obj)']
            self._write_summary_op = tf.summary.merge(summary_merge_list)
            self._summary_writer = tf.summary.FileWriter(self._tboard_save_dir, graph=self._sess.graph)

        LOG.info('Initialize celeba-hq bisenetv2 trainner complete')

    def _compute_warmup_lr(self, warmup_steps, name):
        with tf.variable_scope(name_or_scope=name):
            factor = tf.pow(self._init_learning_rate / self._warmup_init_learning_rate, 1.0 / warmup_steps)
            warmup_lr = self._warmup_init_learning_rate * tf.pow(factor, self._global_step)
        return warmup_lr

    def train(self):
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if CFG['TRAIN']['RESTORE_FROM_SNAPSHOT']['ENABLE']:
            try:
                LOG.info('=> Restoring weights from: {:s} ... '.format(self._initial_weight))
                self._loader.restore(self._sess, self._initial_weight)
                global_step_value = self._sess.run(self._global_step)
                remain_epoch_nums = self._train_epoch_nums - math.floor(global_step_value / self._steps_per_epoch)
                epoch_start_pt = self._train_epoch_nums - remain_epoch_nums
            except OSError as e:
                LOG.error(e)
                LOG.info('=> {:s} does not exist !!!'.format(self._initial_weight))
                LOG.info('=> Now it starts to train BiseNetV2 from scratch ...')
                epoch_start_pt = 1
            except Exception as e:
                LOG.error(e)
                LOG.info('=> Can not load pretrained model weights: {:s}'.format(self._initial_weight))
                LOG.info('=> Now it starts to train BiseNetV2 from scratch ...')
                epoch_start_pt = 1
        else:
            LOG.info('=> Starts to train BiseNetV2 from scratch ...')
            epoch_start_pt = 1

        for epoch in range(epoch_start_pt, self._train_epoch_nums):
            train_epoch_losses = []
            train_epoch_mious = []
            #traindataset_pbar = tqdm.tqdm(range(1, self._steps_per_epoch))
            self.batch_data=self.dataloader.get_iter()
            index=0
            for src,label in self.batch_data:
                index+=1
                feed_dict={self.src_input:src,self.label_input:label[:,:,:,:1]}
                if self._enable_miou and epoch % self._record_miou_epoch == 0:
                    _, _,summary, train_step_loss, global_step_val = self._sess.run(
                        fetches=[
                            self._train_op, self._miou_update_op,
                            self._write_summary_op_with_miou,
                            self._loss, self._global_step
                        ],feed_dict=feed_dict

                    )
                    train_step_miou = self._sess.run(
                        fetches=self._miou,feed_dict=feed_dict
                    )
                    train_epoch_losses.append(train_step_loss)
                    train_epoch_mious.append(train_step_miou)
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    # traindataset_pbar.set_description(
                    #     'train loss: {:.5f}, miou: {:.5f}'.format(train_step_loss, train_step_miou)
                    # )
                    sys.stdout.write('Epoch:{}/{}   {}/{} train loss: {:.5f}, miou: {:.5f} \r'.format(epoch,self._train_epoch_nums,index,self._steps_per_epoch,train_step_loss, train_step_miou))
                    sys.stdout.flush()
                else:
                    _,summary, train_step_loss, global_step_val = self._sess.run(
                        fetches=[
                            self._train_op, self._write_summary_op,
                            self._loss, self._global_step
                        ],feed_dict=feed_dict
                    )
                    train_epoch_losses.append(train_step_loss)
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    sys.stdout.write('Epoch:{}/{}   {}/{} train loss: {:.5f}, miou: {:.5f} \r'.format(epoch,self._train_epoch_nums,index,self._steps_per_epoch,train_step_loss, train_step_miou))
                    sys.stdout.flush()

            train_epoch_losses = np.mean(train_epoch_losses)
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                train_epoch_mious = np.mean(train_epoch_mious)

            if epoch % self._snapshot_epoch == 0:
                if self._enable_miou:
                    snapshot_model_name = 'celebamaskhq_train_miou={:.4f}.ckpt'.format(train_epoch_mious)
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                    os.makedirs(self._model_save_dir, exist_ok=True)
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch)
                else:
                    snapshot_model_name = 'celebamaskhq_train_loss={:.4f}.ckpt'.format(train_epoch_losses)
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                    os.makedirs(self._model_save_dir, exist_ok=True)
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch)

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                LOG.info('=> Epoch: {:d} Time: {:s} Train loss: {:.5f} Train miou: {:.5f} ...'.format(epoch, log_time,train_epoch_losses,train_epoch_mious,))
            else:
                LOG.info('=> Epoch: {:d} Time: {:s} Train loss: {:.5f} ...'.format(epoch, log_time,train_epoch_losses,))
        LOG.info('Complete training process good luck!!')
        return

if __name__=='__main__':
    bisenet=BiseNetV2CelebamaskhqTrainer()
    bisenet.train()

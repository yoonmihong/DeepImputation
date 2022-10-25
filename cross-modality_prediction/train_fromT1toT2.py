import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from model.UnetGAN_percetual_loss import UnetGAN
from datetime import datetime
import logging
import load as loader
import random
import numpy as np
import lib.metrics as metrics
import time
tf_summary = tf.compat.v1.summary

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


tf.compat.v1.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')

# IBIS HR
tf.compat.v1.flags.DEFINE_integer('image_size_z', 144, 'image size z dimension')
tf.compat.v1.flags.DEFINE_integer('image_size_y', 192, 'image size y dimension')
tf.compat.v1.flags.DEFINE_integer('image_size_x', 160, 'image size x dimension')


tf.compat.v1.flags.DEFINE_bool('use_lsgan', True,
					 'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.compat.v1.flags.DEFINE_string('norm', 'instance',
					   '[instance, batch] use instance norm or batch norm, default: instance')
tf.compat.v1.flags.DEFINE_float('learning_rate', 1e-4,
					  'initial learning rate for Adam, default: 0.0002')
tf.compat.v1.flags.DEFINE_float('beta1', 0.5,
					  'momentum term of Adam, default: 0.5')
tf.compat.v1.flags.DEFINE_integer('ngf', 64,
						'number of gen filters in first conv layer, default: 64')
tf.compat.v1.flags.DEFINE_integer('early_stop', 20, 'early stop for the number of epochs which does not increase ssim, default: 64')
tf.compat.v1.flags.DEFINE_float('lamda_l1', 25.0,
					  'coefficient of l1 norm, default: 100')
tf.compat.v1.flags.DEFINE_float('beta_cor', 0.0,
					  'coefficient of l1 norm, default: 0.0')
tf.compat.v1.flags.DEFINE_float('lamda_p', 25.0,
					  'coefficient of l1 norm, default: 0.0')
tf.compat.v1.flags.DEFINE_string("GPU", "0", "GPU to use")
# IBIS
tf.compat.v1.flags.DEFINE_string('paired_image_path',
					   '/Human2/MachineLearningStudies/Imputation/ymhong/code/DeepImputation/cross-modality_prediction/pickle/paired_EBDS_resampled_1year/',
					   'where pickcle data saved. format:[x,y,path]')

tf.compat.v1.flags.DEFINE_string('load_model', None,
					   'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.compat.v1.flags.DEFINE_integer('start_step', 0,
						'number of gen filters in first conv layer, default: 0')
tf.compat.v1.flags.DEFINE_string('weight_dir', '/ASD/Autism/IBIS2/IBIS_DL_Prediction/Code/model_weights/unet_3d.pkl',
					   'the dir of weight file for fine tune')
tf.compat.v1.flags.DEFINE_string('checkpoints_dir', '/Human2/MachineLearningStudies/Imputation/ymhong/code/DeepImputation/cross-modality_prediction/checkpoints/PGAN_T1toT2_EBDS_resampled_1year/',
					   'the dir for checkpoints')
FLAGS = tf.compat.v1.flags.FLAGS


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU

run_config = tf.compat.v1.ConfigProto()	
run_config.gpu_options.allow_growth=True
# IBIS
data_train_t1_t2 = loader.Data(os.path.join(FLAGS.paired_image_path, 'train'), 'train')
data_val_t1_t2 = loader.Data(os.path.join(FLAGS.paired_image_path, 'val'), 'val')
data_test_t1_t2 = loader.Data(os.path.join(FLAGS.paired_image_path, 'test'), 'test')



# start fetching data
data_train_t1_t2.start_thread()
data_val_t1_t2.start_thread()
data_test_t1_t2.start_thread()

# calculate epoch
epoch = int(data_train_t1_t2.len / FLAGS.batch_size)
epoch_val = int(data_val_t1_t2.len / FLAGS.batch_size)
epoch_test = int(data_test_t1_t2.len / FLAGS.batch_size)


def get_checkpoint_dir():
	current_time = datetime.now().strftime("%Y%m%d-%H%M")
	checkpoints_dir = os.path.join(FLAGS.checkpoints_dir, current_time)

	try:
		os.makedirs(checkpoints_dir)
	except os.error:
		pass

	return checkpoints_dir


def init_writer(checkpoints_dir, sess):
	train_step_summary_writer = tf_summary.FileWriter(os.path.join(checkpoints_dir, 'train_step'), sess.graph)
	train_epoch_summary_writer = tf_summary.FileWriter(os.path.join(checkpoints_dir, 'train_epoch'), sess.graph)
	val_epoch_summary_writer = tf_summary.FileWriter(os.path.join(checkpoints_dir, 'val_epoch'), sess.graph)
	test_epoch_summary_writer = tf_summary.FileWriter(os.path.join(checkpoints_dir, 'test_epoch'), sess.graph)
	
	return train_step_summary_writer, train_epoch_summary_writer, val_epoch_summary_writer, test_epoch_summary_writer


def add_summary(sess, summary_op, feed_dict, summary_writer, step):
	summary_str = sess.run(summary_op, feed_dict)
	summary_writer.add_summary(summary_str, step)
	summary_writer.flush()


def train():
	if FLAGS.load_model is not None:
		checkpoints_dir = os.path.join(FLAGS.checkpoints_dir, FLAGS.load_model)
	else:
		checkpoints_dir = get_checkpoint_dir()
		if not os.path.exists(checkpoints_dir):
			os.makedirs(checkpoints_dir)


	graph = tf.Graph()
	with graph.as_default():
		input_src = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_size_z, FLAGS.image_size_y,
															   FLAGS.image_size_x, 1])
		input_tar = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_size_z, FLAGS.image_size_y,
															   FLAGS.image_size_x, 1])
		ph_ssim = tf.compat.v1.placeholder(tf.float32, name='ssim')
		tf_summary.scalar(name='ssim', tensor=ph_ssim)

		unet_gan = UnetGAN(
			input_src,
			input_tar,
			condition=None,
			weight_dir=FLAGS.weight_dir,
			batch_size=FLAGS.batch_size,
			image_size_z=FLAGS.image_size_z,
			image_size_y=FLAGS.image_size_y,
			image_size_x=FLAGS.image_size_x,
			use_lsgan=FLAGS.use_lsgan,
			norm=FLAGS.norm,
			lamda_l1=FLAGS.lamda_l1,
			beta_cor=FLAGS.beta_cor,
			lamda_p=FLAGS.lamda_p,
			learning_rate=FLAGS.learning_rate,
			beta1=FLAGS.beta1,
			ngf=FLAGS.ngf
		)

		G_loss, D_Y_loss, fake_y = unet_gan.model()

		optimizer_G = unet_gan.optimize_G(G_loss)
		optimizer_D = unet_gan.optimize_D(D_Y_loss)

		summary_op = tf.compat.v1.summary.merge_all()
		saver = tf.compat.v1.train.Saver()

	with tf.compat.v1.Session(graph=graph) as sess:
		if FLAGS.load_model is not None:
			print ("load model...")
			start_step = FLAGS.start_step
			model_name = 'model.ckpt-'+str(start_step)+'.meta'
			meta_graph_path = os.path.join(checkpoints_dir, model_name)
			restore = tf.train.import_meta_graph(meta_graph_path)
			model_name = 'model.ckpt-'+str(start_step)
			restore.restore(sess, os.path.join(checkpoints_dir, model_name))
			checkpoints_dir = os.path.join(FLAGS.checkpoints_dir, FLAGS.load_model)
		else:
			print ("start initializing...")
			sess.run(tf.compat.v1.global_variables_initializer())
			start_step = 0

		summary_dir = os.path.join(checkpoints_dir, 'tensorboard')
		if not os.path.isdir(summary_dir):
			os.mkdir(summary_dir)
		print ("initializing writer...")
		train_step_summary_writer, train_epoch_summary_writer, \
		val_epoch_summary_writer, test_epoch_summary_writer = init_writer(summary_dir, sess)
		print ("finished initialization writer...")
		# initialize measurements
		ssim_per_epoch_train = 0.0
		best_val_ssim = 0.0
		violate_early_stop = 0
		breaker = False

		step = start_step
		while True:
			if step > 30000:
				break
			
			# T1 -> T2
			x_t1, x_t2, subj_id = data_train_t1_t2.next_batch(FLAGS.batch_size)
			if step==0:
				print("training subj id: ", subj_id)

			# normalize intensity range to [-1,1]
			x_t1 = x_t1 * 2 - 1
			x_t2 = x_t2 * 2 - 1


			# train
			feed_dict = {input_src: x_t1, input_tar: x_t2, unet_gan.is_training: True, ph_ssim: 0.} 

			_, G_loss_train, D_Y_loss_train, fake_y_train = sess.run([optimizer_G, G_loss, D_Y_loss, fake_y],
																	 feed_dict=feed_dict)

			# calculate measurement for each batch
			ssim_per_batch_train = metrics.compute_ssim(x_t2, fake_y_train) 
			# calculate measurement for each epoch
			ssim_per_epoch_train += ssim_per_batch_train

			if step % 2 == 0: # ymhong: train discriminator every other step
				_ = sess.run(optimizer_D, feed_dict=feed_dict)
				add_summary(sess, summary_op, feed_dict, train_step_summary_writer, step)


			if step % epoch == 0:
				begin_time=time.time()
				# train
				cur_epoch = int(step / epoch)
				print ("Epoch: ", cur_epoch)
				ssim_per_epoch_train /= float(epoch)

				print('ssim_per_epoch_train')
				print(ssim_per_epoch_train)

				feed_dict[ph_ssim] = ssim_per_epoch_train

				add_summary(sess, summary_op, feed_dict, train_epoch_summary_writer, step)

				ssim_per_epoch_train = 0.0

				

				# val
				# initialize measurements
				val_feed_dict = {}
				ssim_per_epoch_val = 0.0
				for step_val in range(epoch_val):
					image_t1_val, image_t2_val, subj_id = data_val_t1_t2.next_batch(FLAGS.batch_size)
					if step_val==0:
						print("validation subj id: ", subj_id)
					x_val = image_t1_val
					y_val = image_t2_val

  
					x_val = x_val * 2 - 1
					y_val = y_val * 2 - 1

					val_feed_dict = {input_src: x_val, input_tar: y_val, unet_gan.is_training: False}
					fake_y_val = (sess.run(fake_y, feed_dict=val_feed_dict))

					# calculate measurement for each batch
					ssim_per_batch_val = metrics.compute_ssim(y_val, fake_y_val)

					# calculate measurement for each epoch
					ssim_per_epoch_val += ssim_per_batch_val

				ssim_per_epoch_val /= float(epoch_val)

				val_feed_dict[ph_ssim] = ssim_per_epoch_val
				add_summary(sess, summary_op, val_feed_dict, val_epoch_summary_writer, step)

				# test
				# initialize measurements
				test_feed_dict = {}
				ssim_per_epoch_test = 0.0
				for step_test in range(epoch_test):
					image_t1_test, image_t2_test, subj_id = data_test_t1_t2.next_batch(FLAGS.batch_size)
					if step_test==0:
						print("test subj id: ", subj_id)
					x_test = image_t1_test
					y_test = image_t2_test


					x_test = x_test * 2 - 1
					y_test = y_test * 2 - 1

					test_feed_dict = {input_src: x_test, input_tar: y_test, unet_gan.is_training: False}
					fake_y_test = (sess.run(fake_y, feed_dict=test_feed_dict))

					# calculate measurement for each batch
					ssim_per_batch_test = metrics.compute_ssim(y_test, fake_y_test)

					# calculate measurement for each epoch
					ssim_per_epoch_test += ssim_per_batch_test
					
				ssim_per_epoch_test /= float(epoch_test)

				test_feed_dict[ph_ssim] = ssim_per_epoch_test
				add_summary(sess, summary_op, test_feed_dict, test_epoch_summary_writer, step)
				
				elapsed_time=time.time()-begin_time
				print ("Elapsed: {:.2f}s".format(elapsed_time))
				# if step > 10000 and ssim_per_epoch_val > best_val_ssim:
				# 	best_val_ssim = ssim_per_epoch_val
					

				# 	logging.info('-----------Step %d:-------------' % step)
				# 	logging.info('  Best ssim in valset   : {}'.format(best_val_ssim))

				# 	save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
				# 	logging.info("Model saved in file: %s" % save_path)

				if step > 10000:
					if (ssim_per_epoch_val > best_val_ssim):
						best_val_ssim = ssim_per_epoch_test 
						#model_name = 'model.ckpt-'+str(start_step)+'.meta'
						cur_model_path = os.path.join(checkpoints_dir, f'model_{step}.ckpt')
						# saver.save(sess, cur_model_path, global_step=step)
						# logs.add('Save', 'Best_model', f'epoch: {cur_epoch}, save best model to {cur_model_path}\n',
						# 		 empty_line=1)
						logging.info('-----------Step %d:-------------' % step)
						logging.info('  Best ssim in valset   : {}'.format(best_val_ssim))

						save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
						logging.info("Model saved in file: %s" % save_path)
						violate_early_stop = 0

					else:
						violate_early_stop += 1
						logging.info('  Early stop count  : {}'.format(violate_early_stop))
						if violate_early_stop > FLAGS.early_stop:
							# logs.add('Train', 'Early_stop', f'stop epoch: {cur_epoch}, best model at {cur_model_path}',
							# 		 empty_line=1)
							breaker = True
							break
					if breaker == True:
						break
				logging.info('-----------Step %d:-------------' % step)
				logging.info('  G_loss   : {}'.format(G_loss_train))
				logging.info('  D_Y_loss : {}'.format(D_Y_loss_train))
				if breaker == True:
					break
			step +=1
			if breaker == True:
				break
			

def main(unused_argv):
	train()
	
	data_train_t1_t2.stop()
	data_val_t1_t2.stop()
	data_test_t1_t2.stop()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	tf.compat.v1.app.run()

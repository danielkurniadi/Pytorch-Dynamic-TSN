

def print_basic_opts(opts):
	print("------------------- Runtime Configs -----------------")
	print(".. --name %s" %opts.name)
	print(".. --gpu_ids %s" %opts.gpu_ids)
	print(".. --serial_batches %s" %opts.serial_batches)
	print(".. --num_threads %s" %opts.num_threads)
	print(".. --checkpoints_dir %s" %opts.checkpoints_dir)
	
	print("------------------- Model Configs -------------------")
	print(".. --model %s" %opts.model)
	print(".. --n_classes %s" %opts.output_nc)
	print(".. --input_channels %s" %opts.input_nc)
	print(".. --norm %s" %opts.norm)
	print(".. --init_type %s" %opts.init_type)
	print(".. --init_gain %s" %opts.init_gain)

	print("------------------- Input Configs -------------------")
	print(".. --dataset_mode %s" %opts.dataset_mode)
	print(".. --batch_size %s" %opts.batch_size)
	print(".. --input_size %s" %opts.input_size)
	print(".. --input_range %s" %opts.input_means)
	print(".. --input_std %s" %opts.input_std)
	print(".. --preprocess %s" %opts.preprocess)
	print(".. --crop_size %s" %opts.crop_size)
	print(".. --no_flip %s" %opts.no_flip)

	print("------------------ Additional Params ----------------")
	print(".. --load_epoch %s" %opts.load_epoch)
	print(".. --verbose %s" %opts.verbose)
	print(".. --suffix %s" %opts.suffix)


def print_train_opts(opts):
	print("------------------- Runtime Configs -----------------")
	print(".. --save_freq_epoch %s" %opts.save_freq_epoch)
	print(".. --continue_last %s" %opts.continue_last)
	
	print("------------------- Learning Configs -----------------")
	print(".. --epochs %s" %opts.epochs)
	print(".. --momentum %s" %opts.momentum)
	print(".. --lr %s" %opts.lr)
	print(".. --lr_policy %s" %opts.lr_policy)

	print(".. --lr_decay_factor %s" %opts.lr_decay_factor)
	print(".. --lr_decay_iters %s" %opts.lr_decay_iters)
	print(".. --eval_freq_epoch %s" %opts.eval_freq_epoch)
	print(".. --print_freq_iters %s" %opts.print_freq_iters)

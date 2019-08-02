import os
import shlex
from subprocess import Popen, PIPE

from core.utils.file_system import safe_mkdir

dense_flow_exec_file = "build/dense_flow_gpu"
dense_flow_cmd_tmpl = (
	'{0} --vidFile={1} --xFlowFile={flow_x_prefix} --yFlowFile={flow_y_prefix} --imgFile={img_name_tmpl} '
	'--bound={bound} --type={flow_type} --device_id={gpu_id} --step={step} -h {height} -w {width}'
)


def run_video_dense_flow(
	video_path,
	outdir,
	bound = 20,
	flow_type = 1,
	gpu_id = 0,
	step = 10,
	height = 224,
	width = 224
):
	""" Dense Flow for video file input

	Outputs Dense Flow of L frames (x and y) from processing videos input
	and RGB frames each correspond to dense flow frame at time t.
	"""
	def call_proc(cmd):
		""" This runs in separate thread."""
		# p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
		# out, err = p.communicate()
		
		return shlex.split(cmd)

	flow_x_prefix = os.path.join(outdir, 'flow_x_')  # prefix of flow x frames
	flow_y_prefix = os.path.join(outdir, 'flow_y_')  # prefix of flow y frames
	img_name_tmpl = os.path.join(outdir, 'img_')
	safe_mkdir(outdir)  # create directory for each video data

	dense_flow_cmd = dense_flow_cmd_tmpl.format(
		dense_flow_exec_file, video_path, 
		flow_x_prefix=flow_x_prefix, flow_y_prefix=flow_y_prefix,
		img_name_tmpl=img_name_tmpl, bound=bound, 
		flow_type=flow_type, gpu_id=gpu_id, step=step,
		height=height, width=width
	)

	return call_proc(dense_flow_cmd)


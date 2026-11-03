from .common import *
import comfy
import comfy.model_management

def get_torch_device_patched():
	# just do 0 for now. need proof of concept
	return torch.device(f"vulkan:{0}")



def get_free_memory_patched(dev=None, torch_free_too=False):
	global directml_enabled
	if dev is None:
		dev = get_torch_device()
		
	# this is necessary to find the vulkan device at all
	if "vulkan" in str(dev):
		# no idea if torchvulkan exposes memory budget.
		# will have to investigate later
		mem_free_total = 1024 * 1024 * 1024
		mem_free_torch = mem_free_total
		directml_enabled = False
	elif hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
		mem_free_total = comfy.system_memory.virtual_memory_available()
		mem_free_torch = mem_free_total
	else:
		if directml_enabled:
			mem_free_total = 1024 * 1024 * 1024 #TODO
			mem_free_torch = mem_free_total
		elif is_intel_xpu():
			stats = torch.xpu.memory_stats(dev)
			mem_active = stats['active_bytes.all.current']
			mem_reserved = stats['reserved_bytes.all.current']
			mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
			mem_free_torch = mem_reserved - mem_active
			mem_free_total = mem_free_xpu + mem_free_torch
		elif is_ascend_npu():
			stats = torch.npu.memory_stats(dev)
			mem_active = stats['active_bytes.all.current']
			mem_reserved = stats['reserved_bytes.all.current']
			mem_free_npu, _ = torch.npu.mem_get_info(dev)
			mem_free_torch = mem_reserved - mem_active
			mem_free_total = mem_free_npu + mem_free_torch
		elif is_mlu():
			stats = torch.mlu.memory_stats(dev)
			mem_active = stats['active_bytes.all.current']
			mem_reserved = stats['reserved_bytes.all.current']
			mem_free_mlu, _ = torch.mlu.mem_get_info(dev)
			mem_free_torch = mem_reserved - mem_active
			mem_free_total = mem_free_mlu + mem_free_torch
		else:
			stats = torch.cuda.memory_stats(dev)
			mem_active = stats['active_bytes.all.current']
			mem_reserved = stats['reserved_bytes.all.current']
			mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
			mem_free_torch = mem_reserved - mem_active
			mem_free_total = mem_free_cuda + mem_free_torch

	if torch_free_too:
		return (mem_free_total, mem_free_torch)
	else:
		return mem_free_total

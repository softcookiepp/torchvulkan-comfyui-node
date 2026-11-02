import comfy
import comfy.model_management

# Patched version of comfy.model_management.get_torch_device that allows vulkan devices to be treated as default
def get_torch_device_patched():
	global directml_enabled
	global cpu_state
	if directml_enabled:
		global directml_device
		return directml_device
	if cpu_state == CPUState.MPS:
		return torch.device("mps")
	if cpu_state == CPUState.CPU:
		return torch.device("cpu")
	else:
		if True:
			# for now, just see if this works c:
			return torch.device(f"vulkan:0")
		else:
			if is_intel_xpu():
				return torch.device("xpu", torch.xpu.current_device())
			elif is_ascend_npu():
				return torch.device("npu", torch.npu.current_device())
			elif is_mlu():
				return torch.device("mlu", torch.mlu.current_device())
			else:
				return torch.device(torch.cuda.current_device())
comfy.model_management.get_torch_device = get_torch_device_patched

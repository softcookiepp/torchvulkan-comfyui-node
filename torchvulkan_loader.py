from .common import *
import comfy
import comfy.model_patcher
import comfy.sd

from .vulkan_monkeypatch import *

def load_ModelPatcher(device, mp):
	assert isinstance(mp.model, torch.nn.Module)
	
	# move the model to the device
	mp.model.to(device)
	
	# overwrite load device
	mp.load_device = device
	
	# this will at least function, I hope

class TorchVulkanLoader:
	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(self):
		devices = [i for i in range(torchvulkan.device_count())]
		return {
			"required": {
				"device": (devices, {"default": 0})
			},
			"optional": {
				"model": ("MODEL",),
				"clip": ("CLIP",),
				"vae": ("VAE",)
			}
		}
	
	RETURN_TYPES = ("MODEL", "CLIP", "VAE")
	
	FUNCTION = "load_to_torchvulkan"
	
	CATEGORY = "Backends"
	
	def load_to_torchvulkan(self, device, model = None, clip = None, vae = None):
		# only do monkeypatching if node is invoked or whatever
		comfy.model_management.get_torch_device = get_torch_device_patched
		comfy.model_management.get_free_memory = get_free_memory_patched
		
		device = torch.device(f"vulkan:{device}")
		if not model is None:
			if isinstance(model, comfy.model_patcher.ModelPatcher):
				load_ModelPatcher(device, model)
			else:
				raise NotImplementedError
		
		if not clip is None:
			if isinstance(clip, comfy.sd.CLIP):
				load_ModelPatcher(device, clip.patcher)
			else:
				raise NotImplementedError
		if not vae is None:
			if isinstance(vae, comfy.sd.VAE):
				load_ModelPatcher(device, vae.patcher)
			else:
				raise NotImplementedError
		return model, clip, vae
		

from .common import *

class TorchVulkanLoader:
	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(self):
		devices = [i for i in range(torchvulkan.device_count())]
		return {
			"required": {
				"device": (devices, {"default": "default ( local-task:// )"})
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
		raise NotImplementedError
		device = torch.device(f"vulkan:{device}")
		if model: model.to(device)
		if clip: clip.to(device)
		if vae: vae.to(device)
		return model, clip, vae
		

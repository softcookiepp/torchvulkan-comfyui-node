from .common import *
from .torchvulkan_loader import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.
NODE_CLASS_MAPPINGS = {
	"Torchvulkan Loader": TorchVulkanLoader
}


__all__ = ['NODE_CLASS_MAPPINGS']

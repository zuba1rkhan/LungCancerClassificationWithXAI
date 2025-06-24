# Standard library imports
import os
import random
import sys

# Core scientific computing
import numpy as np
import pandas as pd

# Machine learning
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Deep learning - PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Computer vision
from torchvision import transforms, models, utils
import torchvision.transforms.functional as TF

# Scattering transform
from kymatio.torch import Scattering2D

# Explainable AI
import captum
from captum.attr import GradientShap, IntegratedGradients, DeepLift
try:
    from captum.attr import KernelShap
    KERNELSHAP_AVAILABLE = True
except ImportError:
    KERNELSHAP_AVAILABLE = False
    print("Note: KernelShap not available in this Captum version, will use alternatives")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bars
from tqdm import tqdm

# Image processing
from PIL import Image

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"📊 Device available: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# Check critical libraries
critical_libs = {
    'torch': torch.__version__,
    'kymatio': 'installed',
    'captum': captum.__version__,
    'sklearn': 'installed'
}

print("\n📋 Critical libraries status:")
for lib, version in critical_libs.items():
    print(f"   ✅ {lib}: {version}")

print("\n🚀 Ready to start the Lung Cancer Classification project!")
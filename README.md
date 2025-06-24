# Lung Cancer Histopathological Image Classification: CNN vs ScatNet with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI-purple)](https://github.com)

## 🔬 Project Overview

This project presents a comprehensive comparison of **Convolutional Neural Networks (CNNs)** and **Scattering Networks (ScatNets)** for lung cancer histopathological image classification. The study includes a **custom Kernel SHAP implementation** from scratch to provide explainable AI insights for medical decision-making.

### 🎯 Key Results
- **CNN Performance**: 99.85% test accuracy
- **ScatNet Performance**: 90.45% test accuracy  
- **Clinical Validation**: Both models exceed 70% clinical requirement
- **XAI Validation**: Custom Kernel SHAP with 99.8% efficiency property

## 📊 Dataset

**Source**: Kaggle Lung Cancer Histopathological Images Dataset
- **Total Images**: 10,000 (5,000 adenocarcinoma + 5,000 benign)
- **Image Format**: H&E stained tissue samples, 224×224 RGB
- **Split**: 80% training (8,000) / 20% testing (2,000)
- **Classes**: Binary classification (Adenocarcinoma vs Benign)

## 🏗️ Architecture Comparison

### CNN Architecture
```
Input (3×224×224)
├── Conv2D(3→32) + BatchNorm + ReLU + MaxPool
├── Conv2D(32→64) + BatchNorm + ReLU + MaxPool  
├── Conv2D(64→128) + BatchNorm + ReLU + MaxPool
├── Conv2D(128→256) + BatchNorm + ReLU + MaxPool
└── Classifier: FC(50,176→512→256→2) + Dropout
```

### ScatNet Architecture
```
Input (3×224×224)
├── Scattering Transform (J=3, L=8)
│   ├── Fixed Morlet Wavelets
│   ├── Multi-scale Analysis (3 scales)
│   └── Multi-orientation (8 orientations)
├── Feature Maps: ~170,128 coefficients
└── Classifier: FC(170,128→512→256→2) + Dropout
```

## 🧠 Custom Kernel SHAP Implementation

### Key Features
- **Superpixel Segmentation**: 16×16 blocks for computational efficiency
- **Coalition Sampling**: 50 random feature combinations
- **Shapley Kernel Weighting**: Proper game-theoretic attribution
- **Medical Validation**: Efficiency property verification
- **Spatial Attribution**: Pixel-level explanatory maps

### Validation Metrics
| Model | Efficiency Property | Captum Correlation | Processing Time |
|-------|-------------------|-------------------|-----------------|
| CNN | 99.8% ± 0.2% | 78% | ~0.3s per image |
| ScatNet | 99.6% ± 0.4% | 77% | ~5s per image |

## 📈 Results Summary

### Performance Comparison
| Model | Test Accuracy | Test F1 | CV Accuracy | CV F1 |
|-------|--------------|---------|-------------|--------|
| **CNN** | **99.85%** | **99.85%** | **99.74%** | **99.74%** |
| **ScatNet** | 90.45% | 90.40% | 83.45% | 82.81% |

### Clinical Metrics
| Model | False Negative Rate | False Positive Rate | Clinical Suitability |
|-------|-------------------|-------------------|---------------------|
| **CNN** | 0.3% (3/1000) | 0.0% (0/1000) | ✅ Primary Screening |
| **ScatNet** | 16.6% (166/1000) | 2.5% (25/1000) | ⚠️ Secondary Validation |

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision kymatio captum scikit-learn matplotlib seaborn pandas numpy tqdm
```

### Dataset Setup
1. Download the lung cancer dataset from Kaggle
2. Extract to project directory:
```
lung_cancer_dataset/
├── adenocarcinoma/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── benign/
    ├── image1.png
    ├── image2.png
    └── ...
```

### Run Complete Pipeline
```python
# Execute all cells sequentially in Jupyter notebook
# main.ipynb
```

## 🔬 Key Technical Innovations

### 1. Architecture Comparison Framework
- **Fair Comparison**: Identical classifier heads for both models
- **Rigorous Validation**: 5-fold cross-validation with stratified sampling
- **Multiple Metrics**: Accuracy, F1, precision, recall, confusion matrices

### 2. Custom Kernel SHAP Implementation
- **From Scratch**: Complete implementation without external SHAP libraries
- **Medical Adaptation**: Superpixel approach optimized for histopathological images
- **Validation Suite**: Efficiency property, correlation analysis, medical relevance

### 3. Clinical Integration
- **Medical Thresholds**: Performance evaluation against clinical requirements
- **Error Analysis**: Detailed false positive/negative assessment
- **Deployment Readiness**: Practical considerations for clinical adoption

## 📊 Experimental Methodology

### Training Protocol
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 16
- **Epochs**: 15-20 with early stopping
- **Regularization**: Dropout (0.5, 0.3), BatchNorm
- **Augmentation**: Random flips, rotation (±20°), color jittering

### Evaluation Framework
- **Cross-Validation**: 5-fold stratified sampling
- **Test Set**: Independent 20% holdout
- **Metrics**: Standard classification + medical-specific
- **Statistical Testing**: Performance significance analysis

## 🏥 Clinical Applications

### Primary Use Cases
1. **Automated Screening**: High-throughput initial diagnosis
2. **Quality Assurance**: Second-opinion validation
3. **Educational Training**: Medical student instruction
4. **Research Tool**: Consistent diagnostic standards

### Deployment Considerations
- **CNN Recommendation**: Primary screening tool (99.85% accuracy)
- **ScatNet Application**: Quality assurance and interpretable analysis
- **Hybrid Approach**: Complementary deployment for enhanced confidence

## 🔍 Explainable AI Insights

### Attribution Pattern Analysis
- **CNN Focus**: Morphological features, cellular atypia
- **ScatNet Focus**: Texture patterns, high-frequency details
- **Medical Relevance**: Both identify diagnostically relevant regions
- **Complementary Information**: Different architectural perspectives

### Validation Results
- **Efficiency Property**: Near-perfect SHAP validity (>99.5%)
- **Spatial Correlation**: High agreement with established libraries
- **Processing Speed**: Real-time capability for clinical use
- **Medical Alignment**: Attribution patterns match pathological indicators

## 📚 Research Contributions

### 1. Scientific Advances
- First systematic CNN vs ScatNet comparison on lung histopathology
- Novel medical-adapted Kernel SHAP implementation
- Comprehensive architectural paradigm analysis
- Clinical validation framework for medical AI

### 2. Technical Innovations
- Custom XAI implementation with full validation suite
- Fair comparison methodology for diverse architectures
- Medical-specific evaluation metrics and thresholds
- Practical deployment guidelines

### 3. Clinical Impact
- Evidence-based model selection for medical applications
- Interpretable AI framework for clinical trust
- Performance benchmarks for medical AI development
- Practical guidelines for healthcare integration

## 📈 Future Directions

### Short-term Enhancements
- [ ] Multi-class cancer classification (squamous cell, large cell)
- [ ] Additional XAI methods (GradCAM, LIME, Integrated Gradients)
- [ ] Ensemble approaches combining CNN and ScatNet
- [ ] Real-time inference optimization

### Long-term Research
- [ ] Multi-institutional validation studies
- [ ] Prospective clinical trial evaluation
- [ ] Integration with hospital information systems
- [ ] Expert pathologist validation of XAI results

### Areas for Contribution
- Additional XAI methods implementation
- Performance optimization
- Dataset expansion
- Clinical validation studies

## 📞 Contact

**Muhammad Zubair Ahmed Khan**  
MSc in Artificial Intelligence  
University of Verona  
📧 Email: [zubairkhan1997@gmail.com]  
🔗 LinkedIn: [https://www.linkedin.com/in/zuba1rkhan]

## 🙏 Acknowledgments

- **University of Verona** - MSc in Artificial Intelligence Program
- **Visual Intelligence Course** - Academic supervision and guidance
- **Kaggle Community** - Lung cancer histopathological dataset
- **PyTorch Team** - Deep learning framework
- **Kymatio Developers** - Scattering transform implementation
- **Captum Team** - Explainable AI library for validation

---

**⭐ Star this repository if you find it useful for your research!**

*This project demonstrates the application of advanced AI techniques to critical medical challenges, bridging the gap between theoretical computer science and practical healthcare solutions.*

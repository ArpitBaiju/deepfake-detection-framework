export const MODEL_CONFIGS = {
  basic_cnn: {
    id: 'basic_cnn',
    name: 'Basic CNN',
    role: 'Baseline',
    roleBadgeColor: 'bg-slate-100 text-slate-700 border-slate-200',
    shortDescription: 'Baseline convolutional neural network with sequential feature extraction.',
    detailedDescription: 'Four convolutional blocks with max pooling and a single binary classification head.',
    architecture: '4-layer Conv2D + MaxPool2D + Linear',
    keyFeatures: ['Lightweight architecture', 'Low latency inference'],
    parametersEstimated: '~3.2M parameters',
    inputResolution: '256 × 256 RGB',
    validationMetrics: {
      accuracy: 93.9,
      auc: 0.966,
      f1Score: 0.926,
      isMeasured: true,
    },
  },

  resnet18: {
    id: 'resnet18',
    name: 'ResNet18',
    role: 'Residual CNN',
    roleBadgeColor: 'bg-slate-100 text-slate-700 border-slate-200',
    shortDescription: 'Uses residual skip connections to improve deep-network training and gradient flow.',
    detailedDescription: 'Standard 18-layer residual network architecture with skip connections bypassing layer pairs.',
    architecture: 'ResNet-18 (torchvision)',
    keyFeatures: ['Residual connections', 'Stable deep training'],
    parametersEstimated: '~11.2M parameters',
    inputResolution: '256 × 256 RGB',
    validationMetrics: {
      accuracy: 97.4,
      auc: 0.997,
      f1Score: 0.969,
      isMeasured: true,
    },
  },

  efficientnet_b0: {
    id: 'efficientnet_b0',
    name: 'EfficientNet-B0',
    role: 'Efficient architecture',
    roleBadgeColor: 'bg-slate-100 text-slate-700 border-slate-200',
    shortDescription: 'Balances accuracy and computational efficiency via compound scaling.',
    detailedDescription: 'Inverted residual MBConv blocks with squeeze-and-excitation optimization.',
    architecture: 'EfficientNet-B0 (torchvision)',
    keyFeatures: ['Compound scaling', 'Squeeze-and-excitation'],
    parametersEstimated: '~5.3M parameters',
    inputResolution: '256 × 256 RGB',
    validationMetrics: {
      accuracy: 94.7,
      auc: 0.999,
      f1Score: 0.935,
      isMeasured: true,
    },
  },

  xception: {
    id: 'xception',
    name: 'Xception',
    role: 'Advanced CNN',
    roleBadgeColor: 'bg-slate-100 text-slate-700 border-slate-200',
    shortDescription: 'Uses depthwise separable convolutions and is computationally heavier.',
    detailedDescription: 'Depthwise separable convolutions that decouple spatial and cross-channel correlation mapping.',
    architecture: 'Legacy Xception (timm)',
    keyFeatures: ['Depthwise separable convolutions', 'Decoupled spatial filters'],
    parametersEstimated: '~22.9M parameters',
    inputResolution: '256 × 256 RGB',
    validationMetrics: {
      accuracy: 95.6,
      auc: 0.997,
      f1Score: 0.947,
      isMeasured: true,
    },
  },
};

export const MODEL_LIST = Object.values(MODEL_CONFIGS);
export const DEFAULT_MODEL_ID = 'basic_cnn';

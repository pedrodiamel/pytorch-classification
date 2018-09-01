# Pytorch Classification

## Training

    bash train-[dataset].sh


## Installation 

    $git clone https://github.com/pedrodiamel/pytorchvision.git
    $cd pytorchvision
    $python setup.py install

### Training visualize

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

    # First install Python server and client 
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server -env_path runs/visdom/
    # http://localhost:8097/



## Accuracy
| Model             | CIFAR10     |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              |        |
| [ResNet18](https://arxiv.org/abs/1512.03385)          |        |
| [ResNet50](https://arxiv.org/abs/1512.03385)          |        |
| [ResNet101](https://arxiv.org/abs/1512.03385)         |        |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       |        |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  |        |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  |        |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       |        |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    |95.36%  |
| [DPN92](https://arxiv.org/abs/1707.01629)             |        |



## Ref
- https://github.com/kuangliu/pytorch-cifar
- https://github.com/Cadene/pretrained-models.pytorch
- http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html


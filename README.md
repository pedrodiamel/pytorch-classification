# Pytorch Classification

This project was created to train a single classification model.

## Training

```bash
python helper/train.py +configs=<CONFIG>
```

## Installation

```bash
$git clone https://github.com/pedrodiamel/pytorchvision.git
$cd pytorchvision
$python setup.py install
```

### Training visualize

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

```bash
# First install Python server and client
pip install visdom
# Start the server (probably in a screen or tmux)
python -m visdom.server -env_path out/runs/visdom/ -port 6006
# http://localhost:6006/
```

For jupyter notebook

```bash
jupyter notebook --port 8080 --allow-root --ip 0.0.0.0 --no-browser
```

### Docker

```bash
docker build -f "Dockerfile" -t torchcls:latest .
docker run -ti --privileged --ipc=host --name torchcls-dev -p 8080:8080 -p 6006:6006 -v $DATASETS:/.datasets torchcls:latest /bin/bash
```

### Dockercompose

```bash
docker-compose up --build -d
docker-compose down
docker exec -it torchcls-dev /bin/bash
```

## Benchmarking

### Accuracy

| Model             | CIFAR10     | CIFAR100    | FERp        | Affect      |
| ----------------- | ----------- | ----------- | ----------- | ----------- |
| PreActResNet18    | 95.36%      | 77.02%      |  87.25      | 43.0        |
| PreActResNet34    | 95.72%      | 78.83%      |             |             |

### Accuracy FER problem

| Model             | Ferp(test)        | AffectNet(val)  | Ckp         | Jaffe      | BU3DFE       | Models      |
| ----------------- | ----------------- | --------------- | ----------- | ---------- | ------------ |------------ |
| PreActResNet18    | 82.372            | 26,100          | 55,307      | 36,318     | 39,828       |             |
| FMPNet            | 79,535            | 29,200          | 65,363      | 46,766     | 41,379       |             |
| CVGG              | 84,316            | 31,150          | 66,201      | 46,269     | 42,069       |             |
| ResNet18          | 87,695            | 34,400          | 71,508      | 50,746     | 45,345       |             |
| AlexNet           | 86,038            | 35,075          | 70,670      | 64,401     | 46,379       |             |
| DeXpression       | 79,694            | 31,875          | 51,117      | 44,279     | 37,241       |             |

## References

- <https://github.com/kuangliu/pytorch-cifar>
- <https://github.com/Cadene/pretrained-models.pytorch>
- <http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html>

<!-- ## Acknowledgments -->

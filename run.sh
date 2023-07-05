# drops
CUDA_VISIBLE_DEVICES=2 python main.py --config=configs/cifar10/drops.yaml

# drops with ce
CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/cifar10/drops_ce.yaml

# general
python main.py --config=configs/default.yaml
# Torch vision framework for cifar10 classification

## Training your Model
You can train your model on either the cifar10 dataset or the clothing1mpp dataset. -->

### Train on Cifar10
`python main.py --config=configs/cifar10/default.yaml`




## Customizing your Training Process
Here's how you can customize different components of your training process:

- **Training loop**: Refer to the examples in the `train_loops` directory
- **Model**: Refer to the examples in the `models` directory
- **Dataset**: Refer to the examples in the `loaders` directory
- **Optimizer**: Refer to the examples in the `optimizer_factory.py` file
- **Loss function**: Refer to the examples in the `loss_function_factory.py` file

# Drops Pytorch Private
This project includes implementation of the Drops model in Pytorch. Here you will find how to download the dataset, load the data, and train your own model. 

<!-- ## Getting Started
To start the training process, run the following command:

`python main.py --config=configs/cifar10/drops.yaml` -->


## Dataset
The data for this project can be found in this [Google Drive](https://drive.google.com/drive/folders/1hiBn4wzvgR0_P53fsJ1YrletlPx5qcSv?usp=sharing).

Here are the different types of images found in the dataset: 

- Dress.zip
- Hoodie.zip
- Jacket.zip
- Knitwear.zip
- Shawl.zip
- Shirt.zip
- Suit.zip
- Sweater.zip
- T-shirt.zip
- Underwear.zip
- Vest.zip
- Windbreaker.zip

We also provide a `labels_changeidx.json` file which contains labels to each sample. The .pt file can be used to find the ids of each split. 

- The train set: noisy labels and the noise rate is yet to be figured out. 
- The validation: mostly clean, 
- The test: set we strongly believe that it is clean.

## Data Loading 
You can refer to the `loaders/clothing1mpp.py` file as an example on how to load the data. Simply modify the example code to fit the path of your data and you should be able to load the data.

## Training your Model
You can train your model on either the cifar10 dataset or the clothing1mpp dataset.

### Train on Cifar10
`python main.py --config=configs/cifar10/default.yaml`


### Train on clothing1mpp
`python main.py --config=configs/clothing1mpp/default.yaml`


## Customizing your Training Process
Here's how you can customize different components of your training process:

- **Training loop**: Refer to the examples in the `train_loops` directory
- **Model**: Refer to the examples in the `models` directory
- **Dataset**: Refer to the examples in the `loaders` directory
- **Optimizer**: Refer to the examples in the `optimizer_factory.py` file
- **Loss function**: Refer to the examples in the `loss_function_factory.py` file

### VAE with Pytorch-Lightning

This is inspired from vae-playground. This is an example where we test out vae and conv_vae models with multiple datasets 
like MNIST, celeb-a and MNIST-Fashion datasets.

This also comes with an example streamlit app & deployed at huggingface.


## Model Training

You can train the VAE models by using `train.py` and editing the `config.yaml` file. \
Hyperparameters to change are:
- model_type [vae|conv_vae]
- alpha
- hidden_dim
- dataset [celeba|mnist|fashion-mnist]

There are other configurations that can be changed if required like height, width, channels etc. It also contains the pytorch-lightning configs as well.



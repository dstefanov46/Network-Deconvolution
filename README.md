## Deep Learning 21/22: Final Project

This is a repository for the Final Project in the subject Deep Learning 21/22. It contains the code for the procedure 
*network deconvolution* proposed in the paper [Network Deconvolution](https://arxiv.org/abs/1905.11926), and also 
partial replication of the experiments performed in that work.

### Environment

This project was completed on a **Windows 10 (x64)** machine, and to re-create the environment in which the code was developed,
you need to run:
```
conda env create -f environment.yml
```
The ``environment.yml`` file is part of this repo. And, to start working in the environment, simply run:
```
conda activate dl_final_env
```

### Experiments

To replicate our experiments, you need to run the ```main.py``` file. For instance, if you would like to train the
original ResNeXt-29 architecture (with batch normalization layers) on the CIFAR-10 dataset with the SGD optimizer, and the remainder of the 
hyperparameter settings as we had them in our experiments, you need to run:

```angular2html
python main.py --lr .1 --optimizer SGD --arch resnext --epochs 20 --dataset cifar10  --batch-size 128 --deconv False --block-fc 0 
```

--------

For any suggestions on how to improve this work, please contact me at 
[ds2243@student.uni-lj.si](https://gmail.google.com/inbox/).

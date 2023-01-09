# TCRPPO
This is the implementation of our TCRPPO model. This paper has already been accepted by the conference "Research in Computational Molecular Biology".

Our DRL framework implementation is based on [stable-baseline3](https://github.com/DLR-RM/stable-baselines3).

## Requirements

* python==3.6.13
* pytorch==1.10.2 + cu113
* Stable-baseline3==0.11.0a7
* ERGO https://github.com/IdoSpringer/ERGO



## Installation guide

Download the code and dataset with the command:

```
git clone https://github.com/ninglab/TCRPPO.git
```

With the code available, please also download the [ERGO repository](https://github.com/IdoSpringer/ERGO) under the code/ERGO directory.

Due to the file size limit in github, we cannot upload our separated TCRs dataset from TCRdb. Please download our TCRs dataset on [google drive](https://drive.google.com/drive/folders/1l5Pf50-7sDcKodeIo-VMHRlODu_ruGtM), and move the directory "tcrdb" under the directory "data".

```
mv ./tcrdb ./data/
```



## Training

To train our *TCRPPO* model, run

```
python ./code/tcr_env.py --num_envs 20 --ergo_model <ergo model path> --peptide_path <peptide path> --bad_ratio 0.0 --hidden_dim 256 --latent_dim 128 --gamma 0.90 --path <model path>
```

<code>num_envs</code> : number of environments running in parallel; determined by the cores of CPU.

<code>ergo_model</code> : the path of pre-trained ergo model.

<code>peptide_path</code>: the path of test peptides

<code>bad_ratio</code>: the ratio of difficult initial state

<code>hidden_dim</code>: the dimension of hidden layer

<code>latent_dim</code>: the dimension of latent layer

<code>gamma</code>: the discount factor

<code>path</code>:  the output directory to save the trained *TCRPPO* models.



## Test

To test our *TCRPPO* model, run

```
python ./code/test_RL_tcrs.py --num_envs 4 --out <result file path> --ergo_model <ergo model> --peptides <peptide file path> --rollout 1 --tcrs <test tcr file path> --path <model path>
```

<code>tcrs</code>:  the path of file with testing TCRs

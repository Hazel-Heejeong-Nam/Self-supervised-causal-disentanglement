# SCADI : Self-supervised CAusal DIsentanglement in latent variable models

<img src="https://github.com/Hazel-Heejeong-Nam/Self-supervised-causal-disentanglement/assets/100391059/93560b8b-6556-4ed9-9919-e11a6758fcb0"  width="500">

### Environment setting and Requirements

```
conda create -n scadi python=3.8 -y
conda activate sacdi
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install -U scikit-learn
conda install -c conda-forge imagemagick
```

### Git clone repo

```
git clone https://github.com/Hazel-Heejeong-Nam/Self-supervised-causal-disentanglement.git
```

# Train

### 1. Create pendulum dataset (Yang et al.)
``` bash
python pendulum_dataset.py
```
### 2. Train model
``` bash
python main.py
```
```
Self-supervised-causal-disentanglement
├── results
│      ├── ${model name}
│      |      ├──  epoch0
│      |      │      ├──  A_epoch0.png
│      |      │      ├──  fixed_img.gif
│      |      │      ├──  random_img.gif
│      |      │      └──  reconstructed.png
│      |      ├──  epoch40
│      |      ├──  epoch80
│      |      ├──  ...
│      |      └──  A_final.png
│      ├── ...
│      └── ${MMDDYY}_summary.txt
├── checkpoints
│      ├── ${model name}
│      |      └──  model_trained.pt
│      └── ...
└── ...

```

# Evaluate

### 3. Create evaluation dataset
``` bash
python evalset.py
```

### 4. Evaluate Observer
``` bash
python eval_observation.py
```
```
tensor([0.2373, 0.1229, 0.1112, 0.0797], device='cuda:0',grad_fn=<AbsBackward0>)
factor : length       target : 0       loss : 1.2886 

tensor([0.4170, 0.1543, 0.0613, 0.0176], device='cuda:0',grad_fn=<AbsBackward0>)
factor : light       target : 0       loss : 1.1443 

tensor([0.1152, 0.1590, 0.0667, 0.1096], device='cuda:0',grad_fn=<AbsBackward0>)
factor : pendulum       target : 1       loss : 1.3404 

tensor([0.1036, 0.0864, 0.0104, 0.0023], device='cuda:0',grad_fn=<AbsBackward0>)
factor : shadloc       target : 0       loss : 1.3344 
```

### 5. Evaluate Interpreter
``` bash
python eval_interpretation.py --length ${label length} --light ${label light} --pendulum ${label pendulum} --loc ${label loc} --checkpoint ${checkpoint dir}
```



- SCADI is written based on :
  - <https://github.com/1Konny/Beta-VAE> (Beta-VAE from Higgins et al.)
  - <https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE> (CausalVAE from Yang et al.)
  - <https://github.com/google-research/disentanglement_lib> (Google research disentanglement library)

Contact **hatbi2000@yonsei.ac.kr** for further question. 

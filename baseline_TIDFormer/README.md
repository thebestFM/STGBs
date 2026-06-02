# TIDFormer: Exploiting Temporal and Interactive Dynamics Makes A Great Dynamic Graph Transformer

In this work, we propose TIDFormer, a Transformer-based dynamic graph model that fully exploits temporal and interactive dynamics with an interpretable self-attention mechanisms at interaction level.

## Environments

Thanks to the authors of [DyGLib](https://arxiv.org/abs/2303.13047) for making their project codes publicly available.
Our implementation primarily utilizes code from DyGLib, with specific updates of TIDFormer applied.

[PyTorch 1.8.1](https://pytorch.org/),
[numpy](https://github.com/numpy/numpy),
[pandas](https://github.com/pandas-dev/pandas), and 
[tqdm](https://github.com/tqdm/tqdm).

## Scripts for Dynamic Link Prediction
#### Model Training
* Example of training *TIDFormer* on *MOOC* dataset:
```{bash}
python train_link_prediction.py --dataset_name mooc --model_name TIDFormer --num_runs 5 --gpu 0
```

#### Model Evaluation
Three (i.e., random, historical, and inductive) negative sampling strategies can be used for model evaluation.
* Example of evaluating *TIDFormer* with *random* negative sampling strategy on *MOOC* dataset:
```{bash}
python evaluate_link_prediction.py --dataset_name mooc --model_name TIDFormer --negative_sample_strategy random --num_runs 5 --gpu 0
```

## Citation
```{bibtex}
@inproceedings{peng2025tidformer,
  title={TIDFormer: Exploiting Temporal and Interactive Dynamics Makes A Great Dynamic Graph Transformer},
  author={Peng, Jie and Wei, Zhewei and Ye, Yuhang},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={2245--2256},
  year={2025}
}
```

## Under Construction

Stay tuned.
# Match-Prompt
Match-Prompt: Improving Multi-task Generalization Ability for Neural Text Matching via Prompt Learning (CIKM 2022)

Paper : https://arxiv.org/abs/2204.02725
```
@inproceedings{Match-Prompt,
  author    = {Shicheng Xu and
               Liang Pang and
               Huawei Shen and
               Xueqi Cheng},
  title     = {Match-Prompt: Improving Multi-task Generalization Ability for Neural Text Matching via Prompt Learning},
  booktitle = {In Proceedings of the 2022 Conference on CIKM},
  publisher = {{ACM}},
  year      = {2022},
}
```

## Specialization Stage
```
cd train_continuous_prompt_hp
python cli_all_layer_adhoc_6.py #Train prompt continuous embedding for different tasks
```
## Generalization Stage
```
python mixed_training
```
## Details running tutorial
Tutorial is comming.

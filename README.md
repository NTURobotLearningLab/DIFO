# [NeurIPS'24] DIFO: Diffusion Imitation from Observations

The Official PyTorch implementation of [**DIFO: Diffusion Imitation from Observations (NeurIPS'24)**](https://arxiv.org/pdf/2410.05429).

[Bo-Ruei Huang](https://borueihuang.com),
[Chun-Kai Yang](https://yck1130.github.io),
[Chun-Mao Lai](https://www.mecoli.net),
[Dai-Jie Wu](https://jaywu109.github.io/),
[Shao-Hua Sun](https://shaohua0116.github.io)
<br />
[Robot Learning Lab](https://nturll.netlify.app/about), [National Taiwan University](https://www.ntu.edu.tw/)
<br />
[[`Paper`]](https://arxiv.org/pdf/2410.05429)
[[`Website`]](https://nturobotlearninglab.github.io/DIFO/)

DIFO is a novel framework for imitation learning from observations that combines adversarial imitation learning with inverse dynamics regularization. It enables learning from expert observations without requiring expert actions.

![teaser](https://nturobotlearninglab.github.io/DIFO/static/images/approach/framework.png)

```bibtex
@inproceeding{huang2024DIFO,
  author    = {Huang, Bo-Ruei and Yang, Chun-Kai and Lai, Chun-Mao and Wu, Dai-Jie and Sun, Shao-Hua},
  title     = {Diffusion Imitation from Observation},
  booktitle = {38th Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year      = {2024},
}
```

## Installation

### Environment Setup
- Python 3.10+
- [MuJoCo](https://github.com/deepmind/mujoco) 2.1+ - Physics engine

```bash
conda create -n difo python=3.10 swig
conda activate difo

pip install -r requirements.txt
```

### Wandb Setup

Setup [Weights & Biases](https://wandb.ai/site) by first logging in with `wandb login <YOUR_API_KEY>`.

Alternatively, you can instead log to stdout by setting `log_format_strs = ["stdout"]` in `scripts/ingredients/logging.py`.

## Download Datasets

Download the datasets from the Google Drive to the `datasets/` directory.

```bash
gdown --id 1Bc9pXnJZxgFUhHwJUKE98Mras1TxTC5J -O datasets --folder
```

## Training

We provide the configuration YAML files for training DIFO and other baselines in the `exp_configs/` directory. 

Including 7 tasks:
- `point_maze`: PointMaze
- `ant_maze`: AntMaze
- `walker`: Walker
- `fetch_push`: FetchPush
- `door`: AdroitDoor
- `kitchen`: OpenMicrowave
- `car_racing`: CarRacing (Image-based)

and 11 algorithms:
- `difo`: DIFO
- `difo-na`: DIFO-NA
- `difo-uncond`: DIFO-Uncond
- `bc`: BC
- `bco`: BCO
- `gaifo`: GAIfO
- `AIRLfO`: AIRLfO
- `waifo`: WAILfO
- `ot-lfo`: OT (LfO)
- `iq-lfo`: IQ-Learn (LfO)
- `depo`: DePO

### Wandb Sweep
You can run the training scripts with Wandb sweep with the following commands:

```bash
./scripts/sweep <config_path>

# Example
./scripts/sweep exp_configs/point_maze/difo.yaml
```

### Single Run

If you prefer to run a single experiment in terminal, you can refer the commands and the parameters in the YAML files. For example, to train DIFO on `PointMaze`, you can run the following command:

```bash
python -m scripts.train_adversarial difo with difo sac_il 1d_condition_diffusion_reward point_maze algorithm_kwargs.bce_weight=0.1 reward.net_kwargs.emb_dim=128
```

## Code Attribution

This project builds heavily upon the [imitation](https://github.com/HumanCompatibleAI/imitation) library. All code under the `imitation/` directory is sourced from their project. We deeply appreciate their contributions to the field of imitation learning.

## Acknowledgements

This work builds upon several excellent open-source projects:
- [imitation](https://github.com/HumanCompatibleAI/imitation) for core imitation learning algorithms and infrastructure
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for environment interface
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [D4RL](https://github.com/Farama-Foundation/D4RL) for environments and demonstrations
- [MuJoCo](https://github.com/deepmind/mujoco) for physics simulation

## License

This project is licensed under the MIT License. Key components have the following licenses:
- Code in `imitation/` directory follows the MIT License from [imitation](https://github.com/HumanCompatibleAI/imitation)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceeding{huang2024DIFO,
  author    = {Huang, Bo-Ruei and Yang, Chun-Kai and Lai, Chun-Mao and Wu, Dai-Jie and Sun, Shao-Hua},
  title     = {Diffusion Imitation from Observation},
  booktitle = {38th Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year      = {2024},
}
```

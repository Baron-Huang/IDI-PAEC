# Instance Gradient-entropy Inhibition-optimized Prior-guiding Actively Explainable Clustering for Achieving Pathologist-like Grading
## 🧔 Authors [* is corresponding authors]
- Pan Huang, _Member_, _IEEE_, Mingrui Ma, Lan Wang, Zhiyong Li*, Qin Jin*, _Senior Member_, _IEEE_

## :fire: News
- [2026/02/18] Our manuscript was submitted to _IEEE Transactions on Medical Imaging (IF 9.8)_.



## :rocket: Pipeline
1. Here's a motivation of our **Instance Gradient-entropy Inhibition-optimized Prior-guiding Active Explanation Clustering (IDI-PAEC)** method:

<div align=center>
<img src="https://github.com/Baron-Huang/TicMIL/blob/main/Main_fig/Motivation_for_IGI_PAEC.jpg" style="width:100%; height:100%;">
</div>
   
2. Here's an overview of our **Instance Gradient-entropy Inhibition-optimized Prior-guiding Active Explanation Clustering (IDI-PAEC)** method:

<div align=center>
<img src="https://github.com/Baron-Huang/TicMIL/blob/main/Main_fig/Main_Frame_for_IGI_PAEC.jpg" style="width:80%; height:80%;">
</div>


## :mag: TODO
<font color="red">**We are currently organizing all the code. Stay tuned!**</font>
- [x] training code
- [x] Evaluation code
- [x] Model code
- [ ] Pretrained weights
- [ ] Datasets





## 🛠️ Getting Started

To get started with **IGI-PAEC**, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/Baron-Huang/IGI_PAEC
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```

3. Training on Swin Transformer-S Backbone
```sh
sh IGI_PAEC_main.sh
Modify: --abla_type sota --run_mode train --random_seed ${seed}
```

4. Evaluation
```sh
sh IGI_PAEC_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed}
```

5. Extract features for plots
```sh
sh IGI_PAEC_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --feat_extract
```

6. Interpretability plots
```sh
sh IGI_PAEC_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --bag_weight
```

## :postbox: Contact
If you have any questions, please contact [Dr. Pan Huang](https://scholar.google.com/citations?user=V_7bX4QAAAAJ&hl=zh-CN) (`panhuang@polyu.edu.hk`).

Set up:

- conda create -n \<env name\> python=3.9 -y
- conda activate \<env name\>
- pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
- pip install -r requirements.txt

Download data from here: https://drive.google.com/file/d/1dMVwwQ9AvgM3QCmthkYJNjs6t3T5fZOk/view?usp=sharing

Upzip it, the working dir will look like this:

```bash
.
├── config.yaml
├── data/
├── datamodule/
├── evaluate_crohme.sh
├── models/
├── requirements.txt
├── scripts/
├── train.py
├── utils/
└── utils.py
```

`train.py`: just run `python train.py` for training

`evaluate_crohme.sh`: remember to `chmod +x evaluate_crohme.sh` before using, `./evaluate_crohme.sh \<path to checkpoint\> \<beam size (1 or greedy decode)\> \<batch size\>`

# MSCI: Addressing CLIP's Inherent Limitations for Compositional Zero-Shot Learning

## Model Structure
![项目结构图](./images/structure.png)




## Project Setup and Requirements

To run the project, follow the steps below.

### Install Required Environment

First, install the necessary environment by running the following command:

```bash
pip install -r requirements.txt
```

### Download Backbone Model (ViT-L14)

Next, you need to download the backbone (ViT-L14) model using `wget`. Use the following command:

```bash
cd <VIT_ROOT>
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

### Dataset Download

We conduct experiments on three datasets: Mit-states, Ut-zappos, and C-GQA. Please download these datasets and place them in the `MSCI/code/download_data` directory. Use the links below to download them:

- **Mit-states**: [Mit-states dataset](https://web.mit.edu/phillipi/Public/states_and_transformations/index.html)
- **Ut-zappos**: [Ut-zappos dataset](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/)
- **C-GQA**: [C-GQA dataset](https://github.com/ExplainableML/czsl)

Once downloaded, run the following command to set up the datasets:

```bash
sh download_data.sh
```

## Model Training

### Training in Closed-World Setting

To train the model in the closed-world setting, use the following command:

```bash
python -u train_base.py \
--clip_arch <VIT_ROOT>/ViT-L-14.pt \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--yml_path ./config/msci/<DATASET>.yml \
--num_workers 10 \
--seed 0
```

### Evaluating in Closed-World Setting

To evaluate the model's performance in the closed-world setting, run the following command:

```bash
python -u test_base.py \
--clip_arch <VIT_ROOT>/ViT-L-14.pt \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--yml_path ./config/msci/<DATASET>.yml \
--num_workers 10 \
--seed 0 \
--load_model <SAVE_ROOT>/<DATASET>/val_best.pt
```

### Evaluating in Open-World Setting

To evaluate the model's performance in the open-world setting, we need to compute feasibility scores for all candidate combinations and filter based on these scores. The configuration files for the feasibility scores of each dataset are embedded in the code, allowing you to directly evaluate the model’s performance in the open-world setting with the following command:

```bash
python -u test_base.py \
--clip_arch <VIT_ROOT>/ViT-L-14.pt \
--dataset_path <DATASET_ROOT>/<DATASET> \
--save_path <SAVE_ROOT>/<DATASET> \
--yml_path ./config/msci/<DATASET>-ow.yml \
--num_workers 10 \
--seed 0 \
--load_model <SAVE_ROOT>/<DATASET>/val_best.pt
```

## Notes

1. **Ensure Directories Are Correct**: Before running the commands, verify that the paths to the model files, datasets, and save directories are correctly specified. Replace placeholders like `<CLIP_MODEL_ROOT>`, `<DATASET_ROOT>`, and `<SAVE_ROOT>` with the actual paths.

2. **Check for Dependencies**: Make sure all required libraries and dependencies are correctly installed using the provided `requirements.txt` file. This ensures that the environment is set up for running the experiments smoothly.

3. **Evaluation Configurations**: Make sure to select the correct configuration file based on your dataset (`<DATASET>` in the commands above), whether for the closed-world or open-world setting.

4. **Troubleshooting**: In case of issues with downloading datasets or model weights, check your internet connection or the validity of the provided download links.


## Acknowledgement

Our code references the following projects:

* [DFSP](https://github.com/Forest-art/DFSP)
* [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer)
* [Troika](https://github.com/bighuang624/Troika)











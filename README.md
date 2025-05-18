# MSCI: Addressing CLIP's Inherent Limitations for Compositional Zero-Shot Learning

## Model Structure
![项目结构图](./github_structure.jpg)




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


## 📊 Model Performance Comparison 

### Performance in Closed-World Setting
| Model | Venue | MIT-States S | U | H | AUC | UT-Zappos S | U | H | AUC | C-GQA S | U | H | AUC |
|-------|--------|--------------|----|----|-----|-------------|----|----|-----|----------|----|----|-----|
| CSP | ICLR | 46.6 | 49.9 | 36.3 | 19.4 | 64.2 | 66.2 | 46.6 | 33.0 | 28.8 | 26.8 | 20.5 | 6.2 |
| DFSP | CVPR | 46.9 | 52.0 | 37.3 | 20.6 | 66.7 | 71.7 | 47.2 | 36.9 | 38.2 | 32.9 | 27.1 | 10.5 |
| HPL | IJCAI | 47.5 | 50.6 | 37.3 | 20.2 | 63.0 | 68.8 | 48.2 | 35.0 | 30.8 | 28.4 | 22.4 | 7.2 |
| GIPCOL | WACV | 48.5 | 49.6 | 36.6 | 19.9 | 65.0 | 68.5 | 48.8 | 36.2 | 31.9 | 28.4 | 22.5 | 7.1 |
| Troika | CVPR | 49.0 | _53.0_ | _39.3_ | 22.1 | 66.8 | 73.8 | _54.6_ | _41.7_ | _41.0_ | _35.7_ | _29.4_ | _12.4_ |
| CDS-CZSL | CVPR | **50.3** | 52.9 | 39.2 | _22.4_ | 63.9 | _74.8_ | 52.7 | 39.5 | 38.3 | 34.2 | 28.1 | 11.1 |
| PLID | ECCV | 49.7 | 52.4 | 39.0 | 22.1 | _67.3_ | 68.8 | 52.4 | 38.7 | 38.8 | 33.0 | 27.9 | 11.0 |
| MSCI | IJCAI | _50.2_ | **53.4** | **39.9** | **22.8** | **67.4** | **75.5** | **59.2** | **45.8** | **42.4** | **38.2** | **31.7** | **14.2** |




### Performance in Open-World Setting
| Model | Venue | MIT-States S | U | H | AUC | UT-Zappos S | U | H | AUC | C-GQA S | U | H | AUC |
|--------|--------|--------------|----|----|-----|-------------|----|----|-----|----------|----|----|-----|
| CSP | ICLR | 46.3 | 15.7 | 17.4 | 5.7 | 64.1 | 44.1 | 38.9 | 22.7 | 28.7 | 5.2 | 6.9 | 1.2 |
| DFSP | CVPR | 47.5 | 18.5 | 19.3 | 6.8 | 66.8 | 60.0 | 44.0 | 30.3 | 38.3 | 7.2 | 10.4 | 2.4 |
| HPL | IJCAI | 46.4 | 18.9 | 19.8 | 6.9 | 63.4 | 48.1 | 40.2 | 24.6 | 30.1 | 5.8 | 7.5 | 1.4 |
| GIPCOL | WACV | 48.5 | 16.0 | 17.9 | 6.3 | 65.0 | 45.0 | 40.1 | 23.5 | 31.6 | 5.5 | 7.3 | 1.3 |
| Troika | CVPR | 48.8 | 18.7 | 20.1 | 7.2 | 66.4 | 61.2 | 47.8 | _33.0_ | _40.8_ | 7.9 | 10.9 | _2.7_ |
| CDS-CZSL | CVPR | **49.4** | **21.8** | **22.1** | **8.5** | 64.7 | _61.3_ | _48.2_ | 32.3 | 37.6 | _8.2_ | _11.6_ | _2.7_ |
| PLID | ECCV | 49.1 | 18.7 | 20.4 | 7.3 | **67.6** | 55.5 | 46.6 | 30.8 | 39.1 | 7.5 | 10.6 | 2.5 |
| MSCI | IJCAI | _49.2_ | _20.6_ | _21.2_ | _7.9_ | _67.4_ | **63.0** | **53.2** | **37.3** | **42.0** | **10.6** | **13.7** | **3.8** |

> **Notes**:
> - **S / U / H**: Seen / Unseen / Harmonic Mean
> - **AUC**: Area Under Curve  
> - **Bold**: Best result  
> - _Italic_: Second-best result


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











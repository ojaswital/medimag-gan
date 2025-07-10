## Generative Adversarial Networks for Medical Image Synthesis

#### This project implements **Wasserstein GAN with Gradient Penalty (WGAN-GP)** and **Denoising Diffusion Probabilistic Models (DDPM)** to generate high-resolution **chest X-ray images** from the RSNA Pneumonia Detection dataset. It includes support for visualization, statistical evaluation, and configurable training using YAML files.  
    Tech Stack: PyTorch, TorchMetrics, FID/KID, NumPy, Matplotlib, DICOM
---

### Project Structure

```
Deep-Learning/
└── Generative_AI/
    ├── README.md          # library versions  
    ├── config/                  # YAML/JSON configs  
    │   ├── config_gan.yaml  
    │   └── config_diffusion.yaml   
    ├── src/                     
    │   ├── wgan/                # WGAN GP‐specific code   
    │   │   ├── model.py         # Generator256, Discriminator, etc.  
    │   │   ├── train.py         # train_wgan_gp() and helpers  
    │   │   └── evaluate.py      
    │   └── diffusion/           # diffusion‐model code   
    │       ├── model.py         # UNet256, Diffusion class  
    │       ├── train.py         # train_diffusion_model()  
    │       └── evaluate.py  
    ├── run_main_train.py        # trains model 
    └── run_main_evaluate.py     # FID/KID/KS/EMD evaluation  


```

---

### Features

* **WGAN-GP** and **DDPM** training pipelines
* Native DICOM support (no PNG conversion needed)
* Evaluation: FID, KID, KS-statistic, Wasserstein Distance
* Pixel distribution plots and Q-Q comparison
* Modular, reproducible, and YAML-configurable

---

### Dataset

* **RSNA Pneumonia Detection Challenge** [[Data](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)] [[Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview)]
* Format: DICOM images
* Preprocessing: Resize → Normalize → Tensor

To use your own data, place `.dcm` files in a directory and update `config.yaml`.

---

### Requirements

Key packages:

* `torch`, `torchvision`
* `pydicom`
* `scipy`, `matplotlib`, `pandas`, `numpy`
* `torchmetrics` (for FID & KID)

---

###  Configuration

Edit the `configs/config_wgan.yaml` or `configs/configs_diffusion.yaml` to change hyperparameters:


---

###  Usage

#### Train WGAN-GP or DDPM

```bash
python run_main_train.py --config configs/config_wgan.yaml
python run_main_train.py --config configs/config_diffusion.yaml
```

#### Evaluate Model

```bash
python run_main_evaluate.py --config configs/config_wgan.yaml
python run_main_evaluate.py --config configs/config_diffusion.yaml
```

---

### Sample Output

* Metric CSV: `FID`, `KID`, `KS`, `EMD`
* Plot: Pixel Intensity Histogram (Real vs Fake)
* Plot: Q–Q Distribution Plot
* Sample Image Grid

---

### Future Work

* [ ] Add UNet-based discriminator
* [ ] Support 3D DICOM volumes
* [ ] Conditional GANs (e.g., pneumonia labels)
* [ ] Train pre-trained GANs (e.g. StyleGAN2)
---
### References
* Gulrajani et al., “Improved Training of Wasserstein GANs,” arXiv preprint arXiv:1704.00028, 2017.
[[Link](arXiv:1704.00028)]
* Karras et al., “Training Generative Adversarial Networks with Limited Data,” arXiv preprint arXiv:2006.11239, 2020.
[[Link](arXiv:2006.11239)]



---
###  Author
Ojaswita Lokre  
[LinkedIn](https://www.linkedin.com/in/ojaswita-lokre-a77031159/)  
[Google Scholar](https://scholar.google.com/citations?user=Y6kAyBEAAAAJ&hl=en&oi=ao)
---

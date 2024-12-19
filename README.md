
<!-- PROJECT LOGO -->


**DeepLIIF** is deployed as a free publicly available cloud-native platform (https://deepliif.org) with Bioformats (more than 150 input formats supported) and MLOps pipeline. We also release **DeepLIIF** implementations for single/multi-GPU training, Torchserve/Dask+Torchscript deployment, and auto-scaling via Pulumi (1000s of concurrent connections supported); details can be found in our [documentation](https://nadeemlab.github.io/DeepLIIF/). **DeepLIIF** can be run locally (GPU required) by [pip installing the package](https://github.com/nadeemlab/DeepLIIF/edit/main/README.md#installing-deepliif) and using the deepliif CLI command. **DeepLIIF** can be used remotely (no GPU required) through the https://deepliif.org website, calling the [cloud API via Python](https://github.com/nadeemlab/DeepLIIF/edit/main/README.md#cloud-deployment), or via the [ImageJ/Fiji plugin](https://github.com/nadeemlab/DeepLIIF/edit/main/README.md#imagej-plugin); details for the free cloud-native platform can be found in our [CVPR'22 paper](https://arxiv.org/pdf/2204.04494.pdf).

© This code is made available for non-commercial academic purposes.

![Version](https://img.shields.io/static/v1?label=latest&message=v1.1.11&color=darkgreen)
[![Total Downloads](https://static.pepy.tech/personalized-badge/deepliif?period=total&units=international_system&left_color=grey&right_color=blue&left_text=total%20downloads)](https://pepy.tech/project/deepliif?&left_text=totalusers)

**This is a modified repository of DeepLIIF, with modifications to enable WSI inference using OpenSlide library and produces QuPath compatible outputs. Please reference original repository for more information**

## Prerequisites
1. Python 3.8

## Installing `deepliif`

DeepLIIF can be `pip` installed:
```shell
$ conda create --name deepliif_env python=3.8
$ conda activate deepliif_env
(deepliif_env) $ conda install -c conda-forge openjdk
(deepliif_env) $ pip install deepliif
(deepliif_env) $ pip install openslide-bin
(deepliif_env) $ pip install openslide-python
(deepliif_env) $ pip install geojson
(deepliif_env) $ pip install pandas
(deepliif_env) $ pip install shapely
```
## WSI Inference
python test_WSI_json.py 

## Support
Please use the [GitHub Issues](https://github.com/nadeemlab/DeepLIIF/issues) tab for discussion, questions, or to report bugs related to DeepLIIF.

## License
© [Nadeem Lab](https://nadeemlab.org/) - DeepLIIF code is distributed under **Apache 2.0 with Commons Clause** license, 
and is available for non-commercial academic purposes. 

## Acknowledgments
This code is inspired by [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Reference
If you find our work useful in your research or if you use parts of this code or our released dataset, please cite the following papers:
```
@article{ghahremani2022deep,
  title={Deep learning-inferred multiplex immunofluorescence for immunohistochemical image quantification},
  author={Ghahremani, Parmida and Li, Yanyun and Kaufman, Arie and Vanguri, Rami and Greenwald, Noah and Angelo, Michael and Hollmann, Travis J and Nadeem, Saad},
  journal={Nature Machine Intelligence},
  volume={4},
  number={4},
  pages={401--412},
  year={2022},
  publisher={Nature Publishing Group}
}

@article{ghahremani2022deepliifui,
  title={DeepLIIF: An Online Platform for Quantification of Clinical Pathology Slides},
  author={Ghahremani, Parmida and Marino, Joseph and Dodds, Ricardo and Nadeem, Saad},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={21399--21405},
  year={2022}
}

@article{ghahremani2023deepliifdataset,
  title={An AI-Ready Multiplex Staining Dataset for Reproducible and Accurate Characterization of Tumor Immune Microenvironment},
  author={Ghahremani, Parmida and Marino, Joseph and Hernandez-Prera, Juan and V. de la Iglesia, Janis and JC Slebos, Robbert and H. Chung, Christine and Nadeem, Saad},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  volume={14225},
  pages={704--713},
  year={2023}
}

@article{nadeem2023ki67validationMTC,
  author = {Nadeem, Saad and Hanna, Matthew G and Viswanathan, Kartik and Marino, Joseph and Ahadi, Mahsa and Alzumaili, Bayan and Bani, Mohamed-Amine and Chiarucci, Federico and Chou, Angela and De Leo, Antonio and Fuchs, Talia L and Lubin, Daniel J and Luxford, Catherine and Magliocca, Kelly and Martinez, Germán and Shi, Qiuying and Sidhu, Stan and Al Ghuzlan, Abir and Gill, Anthony J and Tallini, Giovanni and Ghossein, Ronald and Xu, Bin},
  title = {Ki67 proliferation index in medullary thyroid carcinoma: a comparative study of multiple counting methods and validation of image analysis and deep learning platforms},
  journal = {Histopathology},
  volume = {83},
  number = {6},
  pages = {981--988},
  year = {2023},
  doi = {https://doi.org/10.1111/his.15048}
}

@article{zehra2024deepliifstitch,
  author = {Zehra, Talat and Marino, Joseph and Wang, Wendy and Frantsuzov, Grigoriy and Nadeem, Saad},
  title = {Rethinking Histology Slide Digitization Workflows for Low-Resource Settings},
  journal = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  volume = {15004},
  pages = {427--436},
  year = {2024}
}
```

# RAVE

[![Actions Status](https://github.com/FrancisCardinal/RAVE/actions/workflows/build_test_and_release.yml/badge.svg)](https://github.com/FrancisCardinal/RAVE/actions)

RAVE is a proof of concept aiming to demonstrate the possibility of combining
audio and video processing in hearing aids. RAVE is free and open source.

In the root directory, you will find :

* [A Python library](library/RAVE)
* [Unit tests](library/RAVE/tests)
* The [scripts](.github/workflows) to automate compilation, testing and deployment with [GitHub actions](https://docs.github.com/en/actions)

## Get the code

```bash
# Clone with git in a terminal
git clone https://github.com/FrancisCardinal/RAVE.git
# Go in the root folder
cd RAVE
# Install the dependencies
pip install -r requirements.txt
# Install other dependencies
pip install -i https://test.pypi.org/simple/ --no-deps pyodas-JacobKealey
# Add pre-commit hooks to .git
pre-commit install                                                         
```

## Import models
- Add "saved_model.pth" to RAVE/library/RAVE/src/RAVE/eye_tracker
- Add "resnet18.pth" to RAVE/library/RAVE/src/RAVE/face_detection/verifiers/models/resnet18

## Documentation

* [See here](TODO)

## Improvements

Send us your comments/suggestions to improve the project in [issues](https://github.com/introlab/pyodas/issues).

## Authors

* Anthony Gosselin (@AnthonyGosselin)
* Amélie Rioux-Joyal
* Étienne Deshaies-Samson
* Félix Ducharme Turcotte (@felixducharme1)
* Francis Cardinal (@FrancisCardinal)
* Jacob Kealey (@JacobKealey)
* Jérémy Bélec
* Olivier Bergeron

## License

* [GPLv3](LICENSE)

## Acknowledgments

* François Grondin (@FrancoisGrondin)

![IntRoLab](docs/IntRoLab.png)

[IntRoLab - Laboratoire de robotique intelligente / interactive / intégrée / interdisciplinaire @ Université de Sherbrooke](https://introlab.3it.usherbrooke.ca)

## References
#### For the eye tracker, the code for the conversion of an ellipse to a gaze direction was taken and adapted from [DeepVog](https://github.com/pydsgz/DeepVOG), who also use a [GPLv3](LICENSE) license.

#### Code and pre-trained models for face detection and face recognition used from:

      @article{YOLO5Face,
        title        = {YOLO5Face: Why Reinventing a Face Detector},
        author       = {Delong Qi and Weijun Tan and Qi Yao and Jingfeng Liu},
        booktitle    = {ArXiv preprint ArXiv:2105.12931},
        year         = {2021}
      }
      
      @INPROCEEDINGS{9666941,  
        author       = {Tran, Tan M. and Tran, Nguyen H. and Duong, Soan T. M. and Ta, Huy D. and Nguyen, Chanh D.Tr. and Bui, Trung and Truong, Steven Q.H.},
        booktitle    = {2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)}, 
        title        = {ReSORT: an ID-recovery multi-face tracking method for surveillance cameras},
        year         = {2021}, 
        pages        = {01-08},  
        doi          = {10.1109/FG52635.2021.9666941}
      }
      
      @inproceedings{serengil2020lightface,
        title        = {LightFace: A Hybrid Deep Face Recognition Framework},
        author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
        booktitle    = {2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
        pages        = {23-27},
        year         = {2020},
        doi          = {10.1109/ASYU50717.2020.9259802},
        url          = {https://doi.org/10.1109/ASYU50717.2020.9259802},
        organization = {IEEE}
      }

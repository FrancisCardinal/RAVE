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
# Add pre-commit hooks to .git
pre-commit install                                                         
```

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

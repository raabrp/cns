# Conjugate Natural Selection

This code contains examples for natural gradient descent in a parameter space of
Gaussian distributions over $\mathbb{R}^2$. 

It accompanies a paper by the same name
[paper](https://arxiv.org/abs/2208.13898v4) submitted to arXiv.

Hypothesis class:
$$x \sim \pi(x)$$
where
$$\pi = \mathcal{N}(\mu(\theta), \Sigma(\theta))$$

Parameterization:
$$\theta = \bigg[\mu_x, \mu_y, \sqrt{\Sigma_{xx}}, \Sigma_{xy}, \sqrt{\Sigma_{yy}}\bigg]$$


Inverse Fisher Matrix for parameterization:

``` math
\mathcal{I}^{-1}(\theta) = 
\begin{bmatrix} 
\Sigma_{xx} & 0 & 0 \\
0 & \Sigma_{yy} & 0 \\
0 & 0 & A 
\end{bmatrix}
\quad \text{where} \quad
[A^{-1}]_{ij} =
\frac{1}{2} Tr \bigg( \Sigma^{-1} B[i] \Sigma^{-1}  B[j] \bigg)
```

and

```math
B[1] = \begin{bmatrix}
2 \sqrt{\Sigma_{xx}} & 0 \\ 
0 & 0 
\end{bmatrix}
\quad 
B[2] = \begin{bmatrix} 0 & 1 \\
1 & 0 
\end{bmatrix}
\quad 
B[3] = \begin{bmatrix} 
0 & 0 \\
0 & 2 \sqrt{\Sigma_{yy}}
\end{bmatrix}
```

Gradient estimate:
$$\hat{\nabla} \mathcal{L}(\theta) = \sum_i L(x_i) \log \pi(x_i)$$

Overall update:
$$\theta_{i}(t+1) = \theta_{i}(t) -\eta \sum_{j} [F^{-1}]_{ij} [\hat{\nabla} \mathcal{L}(\theta)]_j$$

where $\eta$ is the learning rate.

# Quick-start

## Installation

* `ffmpeg` must be installed for video output

`requirements.txt` is included for use with `pip`

## Running

```
python cns/main.py
```

# Contributing 

This repository is set up for tight version-control management with
[pyenv](https://github.com/pyenv/pyenv), 
[poetry](https://python-poetry.org/), and
[pre-commit](https://pre-commit.com/)

## Development Environment

Use the right python version and install the dependencies
```
pyenv local 3.11.0
poetry env use 3.11.0
poetry install
```

Install tools for development
```
pre-commit install
```
If they cause too many issue, uninstall with
```
pre-commit uninstall
```

## Environment Activation

The environment can be activated / deactivated with:
* `poetry shell` /  `exit` within the shell
* [direnv](https://direnv.net/)

I use the last option. First, ensure it's setup up correctly 
[for poetry](https://github.com/direnv/direnv/wiki/Python/#poetry), 
then  write an `.envrc` file in the repository root that reads

```
layout poetry
which python
```

followed once by the command to allow automatic activation when the current 
working directory is under the repository root:
```
direnv allow PATH_TO_REPO_ROOT
```

## Tests

``` 
poetry run test
```

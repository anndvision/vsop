# vsop

Implementation of ReLU to the Rescue: Improve Your On-Policy Actor-Critic with Positive Advantages

## installation

```.sh
conda env create -f environment.yml

conda activate vsop

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/

pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -e .
```

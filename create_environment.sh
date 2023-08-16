conda create -n lmd python=3.7
conda activate lmd
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install packaging
pip install tensorflow-gpu==2.9.0 
pip install tensorflow-gan==2.0.0
pip install tensorflow-probability==0.16.0
pip install tqdm
pip install --upgrade "jax[cpu]"
pip install tensorflow_datasets
pip install ml-collections
pip install scikit-image==0.19.2
pip install lpips
pip install scikit-learn==1.0.2
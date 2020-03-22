# Deep Anaglyph 3D

End-to-end generation of good old red-cyan 3D images via CNNs.

### Setup & Usage

Install [pyobjet](https://github.com//MahanFathi/OBJET):

``` sh
git clone https://github.com/MahanFathi/OBJET.git 
cd OBJET
make python -j4
pip install .
```

Install requirements:

``` sh
pip install -r requirements.txt
```

Start training:

``` sh
python main.py --config config.yaml --mode train
```

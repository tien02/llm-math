# Fine tune LLM on Vietnamese Elementary Maths Solving

1. Install dependecies
```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
```

2. Configure the root directory
```
cd scripts
```
Replace `BASE_DIR` environment variable value to the absolute path to the project directory.

3. Training
End-to=end fine tuning
```
bash fine_tune.sh
```
Or using Low Rank Apdaption
```
bash lora_fine_tune.sh
```
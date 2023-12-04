# Fine tune LLM on Vietnamese Elementary Maths Solving

This is the [LoRa checkpoint](https://huggingface.co/tienda02/metamath-mistral7B-lora) which fine tuned [Meta-Math-Mistral-7B](https://huggingface.co/meta-math/MetaMath-Mistral-7B)

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
Replace `BASE_DIR` environment variable value with the absolute path to the project directory.

3. Training
End-to-end fine tuning
```
bash fine_tune.sh
```
Or using [LoRa](https://huggingface.co/docs/peft/conceptual_guides/lora)
```
bash lora_fine_tune.sh
```

4. Inference

Check out this [notebook](./inference.ipynb) for more detail.
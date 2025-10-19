
python 01_train.py --input_size 784 --hidden_size 256 --output_size 10 `
	--epochs 5 --batch 100 --learnrate 0.001 `
	--model "model_vanilla.pt"

python 01_train.py --input_size 784 --hidden_size 256 --output_size 10 `
	--epochs 5 --batch 100 --learnrate 0.001 `
	--noisevar1 1. --noisevar2 2. `
	--model "model_var1p_2p.pt"

python 01_train.py --input_size 784 --hidden_size 256 --output_size 10 `
	--epochs 5 --batch 100 --learnrate 0.001 `
	--noisevar1 2. --noisevar2 1. `
	--model "model_var2p_1p.pt"




#python 01_train.py --input_size 784 --hidden_size 128 --output_size 10 `
#	--epochs 10 --batch 100 --learnrate 0.001 `
#	--model "model_hdn_128.pt"

#foreach ($hidden_size in @(256, 128, 64, 32, 24, 16, 12, 8)) {
foreach ($hidden_size in @(10, 6)) {
	echo "${hidden_size}"
	python 01_train.py --input_size 784 --hidden_size ${hidden_size} --output_size 10 `
		--epochs 10 --batch 100 --learnrate 0.001 `
		--model "model_hdn_${hidden_size}.pt"
}


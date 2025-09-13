
#python .\01_train.py --model ./model_vanilla.pth

python .\01_train_noise.py --model ./model_ns_0p1.pth --rate 0.1
python .\01_train_noise.py --model ./model_ns_0p2.pth --rate 0.2
python .\01_train_noise.py --model ./model_ns_0p3.pth --rate 0.3
python .\01_train_noise.py --model ./model_ns_0p4.pth --rate 0.4

python .\01_train.py --model ./model_drop_0p1.pth --droprate 0.1
python .\01_train.py --model ./model_drop_0p2.pth --droprate 0.2
python .\01_train.py --model ./model_drop_0p3.pth --droprate 0.3
python .\01_train.py --model ./model_drop_0p4.pth --droprate 0.4



#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

foreach ($model_path in @(
    "./models_h64/MNIST_gauss0p0_cp5.pt",
    "./models_h64/MNIST_gauss1p0_cp5.pt",
    "./models_h64/MNIST_gauss2p0_cp5.pt",
    "./models_h64/MNIST_gauss4p0_cp5.pt",
    "./models_h64/MNIST_gauss8p0_cp5.pt",
    "./models_h64/MNIST_gauss16p0_cp5.pt",
    "./models_h64/MNIST_gauss32p0_cp5.pt"
    #"./models_h64/MNIST_pepper0p0_cp5.pt",
    #"./models_h64/MNIST_pepper1p0_cp5.pt",
    #"./models_h64/MNIST_pepper2p0_cp5.pt",
    #"./models_h64/MNIST_pepper4p0_cp5.pt",
    #"./models_h64/MNIST_pepper8p0_cp5.pt",
    #"./models_h64/MNIST_pepper16p0_cp5.pt",
    #"./models_h64/MNIST_pepper32p0_cp5.pt"
    )) {
    echo $model_path
    #$model_path = "./models_h64/MNIST_gauss8p0_cp5.pt"
    #foreach ($ds in @("MNIST", "FashionMNIST")) {
    foreach ($ds in @("MNIST")) {
        #foreach ($infer_noise_type in @("gauss", "pepper")) {
        #foreach ($infer_noise_type in @("gauss")) {
        foreach ($infer_noise_type in @("pepper")) {
            foreach ($infer_noise_std in @(0, 1, 2, 4, 8, 16, 32)) {
            #foreach ($noise_std in @(0, 1)) {
                #echo "ds: ${ds}, infer_noise_type: ${infer_noise_type}, infer_noise_std: ${infer_noise_std}"
                #write-host -nonewline "ds: ${ds}, infer_noise_type: ${infer_noise_type}, infer_noise_std: ${infer_noise_std}, "
                python .\01_run.py --ds ${ds} `
                    --hidden_size 64 `
                    --silent `
                    --model $model_path --infer_noise_type ${infer_noise_type} --infer_noise_std ${infer_noise_std}
            }
        }
    }
}

#foreach ($ds in @("MNIST", "FashionMNIST")) {
##foreach ($ds in @("MNIST")) {
#    foreach ($noise_type in @("gauss", "pepper")) {
#        foreach ($noise_std in @(0, 1, 2, 4, 8, 16, 32)) {
#        #foreach ($noise_std in @(0, 1)) {
#            echo "ds: ${ds}, noise_type: ${noise_type}, noise_std: ${noise_std}"
#            python .\01_run.py --train --epochs 5 --ds ${ds} `
#                --hidden_size 64 `
#                --model models_h64/${ds}.pt --train_noise_type ${noise_type} --train_noise_std ${noise_std}
#        }
#    }
#}

##foreach ($ds in @("MNIST", "FashionMNIST")) {
##foreach ($ds in @("MNIST")) {
#foreach ($ds in @("FashionMNIST")) {
#    foreach ($noise_std in @(0, 1, 2, 4, 8, 16, 32)) {
#    #foreach ($noise_std in @(4, 8, 16, 32)) {
#    #foreach ($noise_std in @(0, 1)) {
#        echo "ds: ${ds}, noise_std: ${noise_std}"
#        python .\01_run.py --train --epochs 80 --ds ${ds} `
#            --hidden_size 64 `
#            --model models_h64/${ds}.pt --noise_std ${noise_std}
#    }
#}


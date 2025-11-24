#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

foreach ($ds in @("MNIST", "FashionMNIST")) {
#foreach ($ds in @("MNIST")) {
    foreach ($noise_type in @("gauss", "pepper")) {
        foreach ($noise_std in @(0, 1, 2, 4, 8, 16, 32)) {
        #foreach ($noise_std in @(0, 1)) {
            echo "ds: ${ds}, noise_type: ${noise_type}, noise_std: ${noise_std}"
            python .\01_run.py --train --epochs 5 --ds ${ds} `
                --hidden_size 64 `
                --model models_h64/${ds}.pt --train_noise_type ${noise_type} --train_noise_std ${noise_std}
        }
    }
}

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


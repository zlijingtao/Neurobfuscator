#Go through the sequence obfuscation
#Step 1. Random Trace Generation
bash tracegen_cifar10.sh
#Step 2. Train LSTM predictors based on Random Architecture-Trace Dataset
bash seq_train_cifar10.sh
#Step 3. Validate LSTM predictors based on Random Architecture-Trace Dataset
bash seq_validate_cifar10.sh
#Step 4. Do Sequence Obfuscation on a CIFAR-10 VGG-11 Architecture
bash seq_obfuscate_cifar10.sh

#Go through the dimension obfuscation
#Step 5. Train Random Forest Regressors based on Random Convolution2D Parameter-Trace Dataset
bash dim_train.sh
#Step 6. Validate Random Forest Regressors based on Random Convolution2D Parameter-Trace Dataset
bash dim_validate.sh
#Step 7. Do Dimension Obfuscation on a 64-128 (Input/Output Channel) Conv2D
bash dim_obfuscate_cifar10.sh

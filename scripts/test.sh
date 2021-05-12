#Go through the sequence obfuscation
#Step 1. Random Trace Generation
bash sq_tracegen_cifar.sh
#Step 2. Train LSTM predictors based on Random Architecture-Trace Dataset
bash sq_train_cifar10.sh
#Step 3. Validate LSTM predictors based on Random Architecture-Trace Dataset
bash sq_validate_cifar10.sh
#Step 4. Do Sequence Obfuscation on a CIFAR-10 VGG-11 Architecture
bash sq_obfuscate.sh

#Go through the parameter obfuscation
#Step 1. Gather Traces of Random Convolution2D operation
bash po_datagen_cifar.sh
#Step 2. Train Random Forest Regressors based on Random Convolution2D Parameter-Trace Dataset
bash po_train.sh
#Step 3. Validate Random Forest Regressors based on Random Convolution2D Parameter-Trace Dataset
bash po_validate.sh
#Step 4. Do Parameter Obfuscation on a 64-128 (Input/Output Channel) Conv2D
bash po_obfuscate_cifar.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
export modelFile=./mobilenet_v1/mobilenet_v1_1.0_224_frozen.pb
export mnnModel=./mobilenet_v1.mnn
export bizCode=mobilenet
./MNNConvert -f TF --modelFile ${modelFile} --MNNModel ${mnnModel} --bizCode ${bizCode}

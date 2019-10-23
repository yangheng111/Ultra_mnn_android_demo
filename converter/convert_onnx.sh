export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
export modelFile=./Mb_Tiny_RFB_FD_320/Mb_Tiny_RFB_FD_train_input_320.onnx
export mnnModel=./Mb_Tiny_RFB_FD_train_input_320.mnn
export bizCode=MNN
./MNNConvert -f ONNX --modelFile ${modelFile} --MNNModel ${mnnModel} --bizCode ${bizCode}

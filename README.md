#Respitrack2_DL
##Info and links
###Keras
* https://stackoverflow.com/questions/60480693/image-augmentation-using-keras-in-python
###Installation
* https://ericzhng.github.io/eric-blogs/2018/10/17/tensorflow-pycharm/
* Install `tensorflow==2.0` !!! (not 2.1) 
##Errors
####Runtime version doesn't support TPU training
use `--runtime-version 1.15`
####AttributeError: 'module' object has no attribute 'v1'
edit the files and remove 'compat.v1' occurences.
more info: https://github.com/tensorflow/models/issues/8081
####HTTP permission denied when executing export_tflite_ssd_graph.py
run `gcloud auth application-default login`
####Bazel run works only in a workspace
run `touch WORKSPACE
####ROCm Configuration Error: Cannot find rocm toolkit path
Use the following instead: 
`
tflite_convert \
--graph_def_file=tflite/tflite_graph.pb \ 
--output_file=tflite/detect.tflite \ 
--output_format=TFLITE \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_dev_values=127 \
--change_concat_input_ranges=false \
--allow_custom_ops
`
source: https://medium.com/@teyou21/convert-a-tensorflow-frozen-graph-to-a-tflite-file-part-3-1ccdb3874c4a
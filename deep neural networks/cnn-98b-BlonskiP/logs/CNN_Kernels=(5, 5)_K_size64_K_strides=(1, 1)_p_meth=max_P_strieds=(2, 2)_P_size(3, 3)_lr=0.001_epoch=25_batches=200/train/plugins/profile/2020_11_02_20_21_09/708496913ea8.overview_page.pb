?  *	?rh???b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateu??&??!??,a??G@)YİØ???1 ??'?D@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatP?2??b??!a??	8@)[{??B??1醄Ȉ=6@:Preprocessing2U
Iterator::Model::ParallelMapV2E??S???!?^m??$@)E??S???1?^m??$@:Preprocessing2F
Iterator::Modelh??HK???!???i}?3@)???B:??1'?e]s"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?S?[ƾ?!??`T@)HP?sׂ?11??h??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice ?8?@d??!?4??T?@) ?8?@d??1?4??T?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Y?$?9??!&??1"I@)?GnM?-q?1??}u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorW?6ɏ?e?!u?Hݸ??)W?6ɏ?e?1u?Hݸ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.?"<
Conv2DBackpropInputConv2DBackpropInputH?-?Z???!H?-?Z???">
Conv2DBackpropFilterConv2DBackpropFilter?s??????!f??7??""
Conv2DConv2D0o??!l?<????",
MaxPoolGradMaxPoolGradj/?Cfʳ?!L??"&
ReluGradReluGradaWQ?????!{'H????""
MatMulMatMul?대??!???^??"$
BiasAddBiasAdd?|??+i??!? ?Q?^??"
ReluRelu??w?$??!:_?H??"$
MaxPoolMaxPoolp:K??ޕ?!?#????",
BiasAddGradBiasAddGradΙf?kݒ?!???????YqG?wD@a⎸#??M@q?tK?K?R@y      Y@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?75.7546% of Op time on the host used eager execution. 100% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.
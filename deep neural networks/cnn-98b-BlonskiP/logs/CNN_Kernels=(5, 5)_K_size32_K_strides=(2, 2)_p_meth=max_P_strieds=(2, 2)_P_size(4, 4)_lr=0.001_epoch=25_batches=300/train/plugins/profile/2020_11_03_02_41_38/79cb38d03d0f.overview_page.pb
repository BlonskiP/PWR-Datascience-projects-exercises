?  *	??~j?4Z@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatB??=Т?!??#Q??A@).???ϡ?1I>4җ@@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??Vд??!Y?Y?YJ3@)??Vд??1Y?Y?YJ3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?l ]l??!?$/?)A@)q??#??1fO???.@:Preprocessing2U
Iterator::Model::ParallelMapV2???!????!&o???x#@)???!????1&o???x#@:Preprocessing2F
Iterator::Model??ʡE???!#?{?6]2@)b??!????1 ?^nA!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Mc{-???!8?V?hT@)?????_z?1î!??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj?Z_$???!?X?88D@)Œr?9>z?1իk?r@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???>9
`?!ګ?????)???>9
`?1ګ?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.?"<
Conv2DBackpropInputConv2DBackpropInput???o???!???o???">
Conv2DBackpropFilterConv2DBackpropFilter?͟?y??!(??????""
Conv2DConv2D@Y?"??!?9?Q5???",
MaxPoolGradMaxPoolGrad ???? ??!?0 M???"&
ReluGradReluGrad??G???!6򝶈??"$
BiasAddBiasAdd4??zꃞ?! Jr
? ??"
ReluRelu?fD?F??!????????"$
MaxPoolMaxPoolǀ???ߘ?!??I?ߑ??""
MatMulMatMul3y?8????!a8ޣ7??",
BiasAddGradBiasAddGrad??k?v??!??jY???Y??????P@a??????@@q?kV/?E@y      Y@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?42.1554% of Op time on the host used eager execution. 100% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.
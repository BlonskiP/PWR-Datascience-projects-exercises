?  *	-?????b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???C?ݢ?!s?????8@)??u6䟡?15?	)?6@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceH?C??ݠ?!W,?ԭ5@)H?C??ݠ?1W,?ԭ5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQ?|a2??!???n?B@)h??????1?? ??/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???v?
??!8??Km?T@)tys?V{??1hrw/@:Preprocessing2U
Iterator::Model::ParallelMapV2?I??ǌ?!??{P?~"@)?I??ǌ?1??{P?~"@:Preprocessing2F
Iterator::Model?^?D??!??J?1@)]Ot]????1?)ES?? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?E?????!?3#??E@)?)s???~?1M?+??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??հ?c?!?C?=???)??հ?c?1?C?=???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.?""
Conv2DConv2D?P:??!?P:??"<
Conv2DBackpropInputConv2DBackpropInput9i?G?W??!?2?˂???">
Conv2DBackpropFilterConv2DBackpropFilter?3?+?[??!O?鰣s??"$
AvgPoolAvgPool????eq??!0Θ)?O??",
AvgPoolGradAvgPoolGrad???Ɋ???!b??_???"&
ReluGradReluGrad??P=??!???cG??"$
BiasAddBiasAdd?ć?W???!?a!*????"
ReluRelu??M??ɞ?!??C)G???",
BiasAddGradBiasAddGradV:?TRk??!??黡+??""
MatMulMatMul??8q?O??!??s????Y?;?P@a???"??@@qߙ?>!?@y      Y@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.1299% of Op time on the host used eager execution. 100% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.
?  *	L7?A`U^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??o?4(??!נ?C:=@)??wcAa??1,?`?]:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateD?!T?٣?!|=/????@)ҧU??f??1c???w8@:Preprocessing2U
Iterator::Model::ParallelMapV2?J??q???!ҡ)??z)@)?J??q???1ҡ)??z)@:Preprocessing2F
Iterator::Model2??n??!???wd7@)?	h"lx??1?]5	N%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?jQLި?!?ׁ
?D@)??????1(㨫?' @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicel??g????!?i?D??@)l??g????1?i?D??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0h!?˷?!?	?&S@)[{C??1??Ãe@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?U??6ol?!Z<???@)?U??6ol?1Z<???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.?"<
Conv2DBackpropInputConv2DBackpropInput???V???!???V???"$
AvgPoolAvgPoolI3?????!8-P?????">
Conv2DBackpropFilterConv2DBackpropFilter???]??!?֗?=???""
Conv2DConv2D??????!^?n????",
AvgPoolGradAvgPoolGrad?]??
??!;??t0??"&
ReluGradReluGrad*??}bQ??!^_gC?u??"$
BiasAddBiasAddo??Er??!???w????""
MatMulMatMulqw&????!A?c0?y??"
ReluRelu}??~ј?!e?XdA@??",
BiasAddGradBiasAddGradT믝%???!x?????Y??????P@a??????@@q??o??IO@y      Y@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?62.5779% of Op time on the host used eager execution. 100% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.
?  *	??(\??^@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate.???!??Ux?9A@)+?`??17'??+/:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat`???8??!O!e?B;@)?s???מ?1-T~׹?8@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?,D?????!?8????S@)??>s֧??129??&@:Preprocessing2U
Iterator::Model::ParallelMapV2?.??[<??!??vs4?&@)?.??[<??1??vs4?&@:Preprocessing2F
Iterator::Model*S?A?њ?!???dp5@)?w??Dg??1???N$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceM?O???!'\u??? @)M?O???1'\u??? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???R???!??.SM?C@)?????z?1}@??Γ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorR???Tj?!i6???@)R???Tj?1i6???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.?""
Conv2DConv2DŖ}*?a??!Ŗ}*?a??"<
Conv2DBackpropInputConv2DBackpropInput?{?MV/??!D?=??H??">
Conv2DBackpropFilterConv2DBackpropFilter_?D?/???!z???YP??"$
AvgPoolAvgPool???U??!¢#??%??",
AvgPoolGradAvgPoolGrad>??r??!R???"???"&
ReluGradReluGrad?G[?????!?E?l??"$
BiasAddBiasAdd?????v??!8????d??"
ReluRelu?T^PÎ??!ݐr?NY??",
BiasAddGradBiasAddGrad{;?????!??h?%???""
MatMulMatMulk????"??!??.?;???Y?;?P@a???"??@@q???3hK@y      Y@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?54.2141% of Op time on the host used eager execution. 100% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.
äÄ
æ³
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*	2.5.0-rc02v1.12.1-53831-ga8b6d5ff93a8¦

input_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
°*#
shared_nameinput_layer/kernel
{
&input_layer/kernel/Read/ReadVariableOpReadVariableOpinput_layer/kernel* 
_output_shapes
:
°*
dtype0
y
input_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*!
shared_nameinput_layer/bias
r
$input_layer/bias/Read/ReadVariableOpReadVariableOpinput_layer/bias*
_output_shapes	
:°*
dtype0

batch_normalization_318/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*.
shared_namebatch_normalization_318/gamma

1batch_normalization_318/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_318/gamma*
_output_shapes	
:°*
dtype0

batch_normalization_318/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*-
shared_namebatch_normalization_318/beta

0batch_normalization_318/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_318/beta*
_output_shapes	
:°*
dtype0

#batch_normalization_318/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*4
shared_name%#batch_normalization_318/moving_mean

7batch_normalization_318/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_318/moving_mean*
_output_shapes	
:°*
dtype0
§
'batch_normalization_318/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*8
shared_name)'batch_normalization_318/moving_variance
 
;batch_normalization_318/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_318/moving_variance*
_output_shapes	
:°*
dtype0
~
dense_318/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
°·*!
shared_namedense_318/kernel
w
$dense_318/kernel/Read/ReadVariableOpReadVariableOpdense_318/kernel* 
_output_shapes
:
°·*
dtype0
u
dense_318/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*
shared_namedense_318/bias
n
"dense_318/bias/Read/ReadVariableOpReadVariableOpdense_318/bias*
_output_shapes	
:·*
dtype0

batch_normalization_319/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*.
shared_namebatch_normalization_319/gamma

1batch_normalization_319/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_319/gamma*
_output_shapes	
:·*
dtype0

batch_normalization_319/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*-
shared_namebatch_normalization_319/beta

0batch_normalization_319/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_319/beta*
_output_shapes	
:·*
dtype0

#batch_normalization_319/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*4
shared_name%#batch_normalization_319/moving_mean

7batch_normalization_319/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_319/moving_mean*
_output_shapes	
:·*
dtype0
§
'batch_normalization_319/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*8
shared_name)'batch_normalization_319/moving_variance
 
;batch_normalization_319/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_319/moving_variance*
_output_shapes	
:·*
dtype0
}
dense_319/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	·*!
shared_namedense_319/kernel
v
$dense_319/kernel/Read/ReadVariableOpReadVariableOpdense_319/kernel*
_output_shapes
:	·*
dtype0
t
dense_319/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_319/bias
m
"dense_319/bias/Read/ReadVariableOpReadVariableOpdense_319/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/input_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
°**
shared_nameAdam/input_layer/kernel/m

-Adam/input_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/input_layer/kernel/m* 
_output_shapes
:
°*
dtype0

Adam/input_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*(
shared_nameAdam/input_layer/bias/m

+Adam/input_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/input_layer/bias/m*
_output_shapes	
:°*
dtype0
¡
$Adam/batch_normalization_318/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*5
shared_name&$Adam/batch_normalization_318/gamma/m

8Adam/batch_normalization_318/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_318/gamma/m*
_output_shapes	
:°*
dtype0

#Adam/batch_normalization_318/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*4
shared_name%#Adam/batch_normalization_318/beta/m

7Adam/batch_normalization_318/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_318/beta/m*
_output_shapes	
:°*
dtype0

Adam/dense_318/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
°·*(
shared_nameAdam/dense_318/kernel/m

+Adam/dense_318/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_318/kernel/m* 
_output_shapes
:
°·*
dtype0

Adam/dense_318/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*&
shared_nameAdam/dense_318/bias/m
|
)Adam/dense_318/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_318/bias/m*
_output_shapes	
:·*
dtype0
¡
$Adam/batch_normalization_319/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*5
shared_name&$Adam/batch_normalization_319/gamma/m

8Adam/batch_normalization_319/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_319/gamma/m*
_output_shapes	
:·*
dtype0

#Adam/batch_normalization_319/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*4
shared_name%#Adam/batch_normalization_319/beta/m

7Adam/batch_normalization_319/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_319/beta/m*
_output_shapes	
:·*
dtype0

Adam/dense_319/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	·*(
shared_nameAdam/dense_319/kernel/m

+Adam/dense_319/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_319/kernel/m*
_output_shapes
:	·*
dtype0

Adam/dense_319/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_319/bias/m
{
)Adam/dense_319/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_319/bias/m*
_output_shapes
:*
dtype0

Adam/input_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
°**
shared_nameAdam/input_layer/kernel/v

-Adam/input_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/input_layer/kernel/v* 
_output_shapes
:
°*
dtype0

Adam/input_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*(
shared_nameAdam/input_layer/bias/v

+Adam/input_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/input_layer/bias/v*
_output_shapes	
:°*
dtype0
¡
$Adam/batch_normalization_318/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*5
shared_name&$Adam/batch_normalization_318/gamma/v

8Adam/batch_normalization_318/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_318/gamma/v*
_output_shapes	
:°*
dtype0

#Adam/batch_normalization_318/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*4
shared_name%#Adam/batch_normalization_318/beta/v

7Adam/batch_normalization_318/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_318/beta/v*
_output_shapes	
:°*
dtype0

Adam/dense_318/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
°·*(
shared_nameAdam/dense_318/kernel/v

+Adam/dense_318/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_318/kernel/v* 
_output_shapes
:
°·*
dtype0

Adam/dense_318/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*&
shared_nameAdam/dense_318/bias/v
|
)Adam/dense_318/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_318/bias/v*
_output_shapes	
:·*
dtype0
¡
$Adam/batch_normalization_319/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*5
shared_name&$Adam/batch_normalization_319/gamma/v

8Adam/batch_normalization_319/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_319/gamma/v*
_output_shapes	
:·*
dtype0

#Adam/batch_normalization_319/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*4
shared_name%#Adam/batch_normalization_319/beta/v

7Adam/batch_normalization_319/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_319/beta/v*
_output_shapes	
:·*
dtype0

Adam/dense_319/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	·*(
shared_nameAdam/dense_319/kernel/v

+Adam/dense_319/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_319/kernel/v*
_output_shapes
:	·*
dtype0

Adam/dense_319/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_319/bias/v
{
)Adam/dense_319/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_319/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ÎB
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*B
valueÿABüA BõA
Î
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
R
'	variables
(trainable_variables
)regularization_losses
*	keras_api

+axis
	,gamma
-beta
.moving_mean
/moving_variance
0	variables
1trainable_variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
ú
:iter

;beta_1

<beta_2
	=decay
>learning_ratemrmsmtmu!mv"mw,mx-my4mz5m{v|v}v~v!v"v,v-v4v5v
f
0
1
2
3
4
5
!6
"7
,8
-9
.10
/11
412
513
F
0
1
2
3
!4
"5
,6
-7
48
59
 
­
		variables
?non_trainable_variables

trainable_variables
@metrics
Alayer_regularization_losses

Blayers
regularization_losses
Clayer_metrics
 
^\
VARIABLE_VALUEinput_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEinput_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
	variables
Dnon_trainable_variables
trainable_variables
Emetrics
Flayer_regularization_losses

Glayers
regularization_losses
Hlayer_metrics
 
 
 
­
	variables
Inon_trainable_variables
trainable_variables
Jmetrics
Klayer_regularization_losses

Llayers
regularization_losses
Mlayer_metrics
 
hf
VARIABLE_VALUEbatch_normalization_318/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_318/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_318/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_318/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
­
	variables
Nnon_trainable_variables
trainable_variables
Ometrics
Player_regularization_losses

Qlayers
regularization_losses
Rlayer_metrics
\Z
VARIABLE_VALUEdense_318/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_318/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
­
#	variables
Snon_trainable_variables
$trainable_variables
Tmetrics
Ulayer_regularization_losses

Vlayers
%regularization_losses
Wlayer_metrics
 
 
 
­
'	variables
Xnon_trainable_variables
(trainable_variables
Ymetrics
Zlayer_regularization_losses

[layers
)regularization_losses
\layer_metrics
 
hf
VARIABLE_VALUEbatch_normalization_319/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_319/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_319/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_319/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
.2
/3

,0
-1
 
­
0	variables
]non_trainable_variables
1trainable_variables
^metrics
_layer_regularization_losses

`layers
2regularization_losses
alayer_metrics
\Z
VARIABLE_VALUEdense_319/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_319/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
­
6	variables
bnon_trainable_variables
7trainable_variables
cmetrics
dlayer_regularization_losses

elayers
8regularization_losses
flayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
.2
/3

g0
h1
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

.0
/1
 
 
 
 
 
 
 
 
 
4
	itotal
	jcount
k	variables
l	keras_api
D
	mtotal
	ncount
o
_fn_kwargs
p	variables
q	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

i0
j1

k	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

p	variables

VARIABLE_VALUEAdam/input_layer/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/input_layer/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_318/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_318/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_318/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_318/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_319/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_319/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_319/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_319/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/input_layer/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/input_layer/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_318/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_318/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_318/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_318/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_319/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_319/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_319/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_319/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

!serving_default_input_layer_inputPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Þ
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputinput_layer/kernelinput_layer/bias'batch_normalization_318/moving_variancebatch_normalization_318/gamma#batch_normalization_318/moving_meanbatch_normalization_318/betadense_318/kerneldense_318/bias'batch_normalization_319/moving_variancebatch_normalization_319/gamma#batch_normalization_319/moving_meanbatch_normalization_319/betadense_319/kerneldense_319/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *.
f)R'
%__inference_signature_wrapper_8288138
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
á
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&input_layer/kernel/Read/ReadVariableOp$input_layer/bias/Read/ReadVariableOp1batch_normalization_318/gamma/Read/ReadVariableOp0batch_normalization_318/beta/Read/ReadVariableOp7batch_normalization_318/moving_mean/Read/ReadVariableOp;batch_normalization_318/moving_variance/Read/ReadVariableOp$dense_318/kernel/Read/ReadVariableOp"dense_318/bias/Read/ReadVariableOp1batch_normalization_319/gamma/Read/ReadVariableOp0batch_normalization_319/beta/Read/ReadVariableOp7batch_normalization_319/moving_mean/Read/ReadVariableOp;batch_normalization_319/moving_variance/Read/ReadVariableOp$dense_319/kernel/Read/ReadVariableOp"dense_319/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp-Adam/input_layer/kernel/m/Read/ReadVariableOp+Adam/input_layer/bias/m/Read/ReadVariableOp8Adam/batch_normalization_318/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_318/beta/m/Read/ReadVariableOp+Adam/dense_318/kernel/m/Read/ReadVariableOp)Adam/dense_318/bias/m/Read/ReadVariableOp8Adam/batch_normalization_319/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_319/beta/m/Read/ReadVariableOp+Adam/dense_319/kernel/m/Read/ReadVariableOp)Adam/dense_319/bias/m/Read/ReadVariableOp-Adam/input_layer/kernel/v/Read/ReadVariableOp+Adam/input_layer/bias/v/Read/ReadVariableOp8Adam/batch_normalization_318/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_318/beta/v/Read/ReadVariableOp+Adam/dense_318/kernel/v/Read/ReadVariableOp)Adam/dense_318/bias/v/Read/ReadVariableOp8Adam/batch_normalization_319/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_319/beta/v/Read/ReadVariableOp+Adam/dense_319/kernel/v/Read/ReadVariableOp)Adam/dense_319/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__traced_save_8288790

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_layer/kernelinput_layer/biasbatch_normalization_318/gammabatch_normalization_318/beta#batch_normalization_318/moving_mean'batch_normalization_318/moving_variancedense_318/kerneldense_318/biasbatch_normalization_319/gammabatch_normalization_319/beta#batch_normalization_319/moving_mean'batch_normalization_319/moving_variancedense_319/kerneldense_319/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/input_layer/kernel/mAdam/input_layer/bias/m$Adam/batch_normalization_318/gamma/m#Adam/batch_normalization_318/beta/mAdam/dense_318/kernel/mAdam/dense_318/bias/m$Adam/batch_normalization_319/gamma/m#Adam/batch_normalization_319/beta/mAdam/dense_319/kernel/mAdam/dense_319/bias/mAdam/input_layer/kernel/vAdam/input_layer/bias/v$Adam/batch_normalization_318/gamma/v#Adam/batch_normalization_318/beta/vAdam/dense_318/kernel/vAdam/dense_318/bias/v$Adam/batch_normalization_319/gamma/v#Adam/batch_normalization_319/beta/vAdam/dense_319/kernel/vAdam/dense_319/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__traced_restore_8288929åä

º

ú
F__inference_dense_318_layer_call_and_return_conditional_losses_8287734

inputs2
matmul_readvariableop_resource:
°·.
biasadd_readvariableop_resource:	·
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
°·*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:·*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
)
¦
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8287774

inputs'
input_layer_8287702:
°"
input_layer_8287704:	°.
batch_normalization_318_8287714:	°.
batch_normalization_318_8287716:	°.
batch_normalization_318_8287718:	°.
batch_normalization_318_8287720:	°%
dense_318_8287735:
°· 
dense_318_8287737:	·.
batch_normalization_319_8287747:	·.
batch_normalization_319_8287749:	·.
batch_normalization_319_8287751:	·.
batch_normalization_319_8287753:	·$
dense_319_8287768:	·
dense_319_8287770:
identity¢/batch_normalization_318/StatefulPartitionedCall¢/batch_normalization_319/StatefulPartitionedCall¢!dense_318/StatefulPartitionedCall¢!dense_319/StatefulPartitionedCall¢#input_layer/StatefulPartitionedCall¬
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinput_layer_8287702input_layer_8287704*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_82877012%
#input_layer/StatefulPartitionedCall
dropout_318/PartitionedCallPartitionedCall,input_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_318_layer_call_and_return_conditional_losses_82877122
dropout_318/PartitionedCallÌ
/batch_normalization_318/StatefulPartitionedCallStatefulPartitionedCall$dropout_318/PartitionedCall:output:0batch_normalization_318_8287714batch_normalization_318_8287716batch_normalization_318_8287718batch_normalization_318_8287720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_828738321
/batch_normalization_318/StatefulPartitionedCallÔ
!dense_318/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_318/StatefulPartitionedCall:output:0dense_318_8287735dense_318_8287737*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_82877342#
!dense_318/StatefulPartitionedCall
dropout_319/PartitionedCallPartitionedCall*dense_318/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_319_layer_call_and_return_conditional_losses_82877452
dropout_319/PartitionedCallÌ
/batch_normalization_319/StatefulPartitionedCallStatefulPartitionedCall$dropout_319/PartitionedCall:output:0batch_normalization_319_8287747batch_normalization_319_8287749batch_normalization_319_8287751batch_normalization_319_8287753*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_828754521
/batch_normalization_319/StatefulPartitionedCallÓ
!dense_319/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_319/StatefulPartitionedCall:output:0dense_319_8287768dense_319_8287770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_82877672#
!dense_319/StatefulPartitionedCallÐ
IdentityIdentity*dense_319/StatefulPartitionedCall:output:00^batch_normalization_318/StatefulPartitionedCall0^batch_normalization_319/StatefulPartitionedCall"^dense_318/StatefulPartitionedCall"^dense_319/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2b
/batch_normalization_318/StatefulPartitionedCall/batch_normalization_318/StatefulPartitionedCall2b
/batch_normalization_319/StatefulPartitionedCall/batch_normalization_319/StatefulPartitionedCall2F
!dense_318/StatefulPartitionedCall!dense_318/StatefulPartitionedCall2F
!dense_319/StatefulPartitionedCall!dense_319/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
ð
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288331

inputs
unknown:
°
	unknown_0:	°
	unknown_1:	°
	unknown_2:	°
	unknown_3:	°
	unknown_4:	°
	unknown_5:
°·
	unknown_6:	·
	unknown_7:	·
	unknown_8:	·
	unknown_9:	·

unknown_10:	·

unknown_11:	·

unknown_12:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *a
f\RZ
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_82877742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
I
-__inference_dropout_318_layer_call_fn_8288406

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_318_layer_call_and_return_conditional_losses_82877122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
*
ñ
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_8288465

inputs6
'assignmovingavg_readvariableop_resource:	°8
)assignmovingavg_1_readvariableop_resource:	°4
%batchnorm_mul_readvariableop_resource:	°0
!batchnorm_readvariableop_resource:	°
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	°*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	°2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	°*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:°*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:°*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:°*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:°2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:°2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:°*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:°2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:°2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:°2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:°2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:°*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:°2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:°2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:°*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:°2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
º

ø
F__inference_dense_319_layer_call_and_return_conditional_losses_8287767

inputs1
matmul_readvariableop_resource:	·-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	·*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs

Ø
9__inference_batch_normalization_318_layer_call_fn_8288478

inputs
unknown:	°
	unknown_0:	°
	unknown_1:	°
	unknown_2:	°
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_82873832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
ù
f
H__inference_dropout_319_layer_call_and_return_conditional_losses_8288516

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
È
g
H__inference_dropout_319_layer_call_and_return_conditional_losses_8287835

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ú?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÆ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*
dtype0*
seed2ÿÿÿÿ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ8>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
¿²

X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288298

inputs>
*input_layer_matmul_readvariableop_resource:
°:
+input_layer_biasadd_readvariableop_resource:	°N
?batch_normalization_318_assignmovingavg_readvariableop_resource:	°P
Abatch_normalization_318_assignmovingavg_1_readvariableop_resource:	°L
=batch_normalization_318_batchnorm_mul_readvariableop_resource:	°H
9batch_normalization_318_batchnorm_readvariableop_resource:	°<
(dense_318_matmul_readvariableop_resource:
°·8
)dense_318_biasadd_readvariableop_resource:	·N
?batch_normalization_319_assignmovingavg_readvariableop_resource:	·P
Abatch_normalization_319_assignmovingavg_1_readvariableop_resource:	·L
=batch_normalization_319_batchnorm_mul_readvariableop_resource:	·H
9batch_normalization_319_batchnorm_readvariableop_resource:	·;
(dense_319_matmul_readvariableop_resource:	·7
)dense_319_biasadd_readvariableop_resource:
identity¢'batch_normalization_318/AssignMovingAvg¢6batch_normalization_318/AssignMovingAvg/ReadVariableOp¢)batch_normalization_318/AssignMovingAvg_1¢8batch_normalization_318/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_318/batchnorm/ReadVariableOp¢4batch_normalization_318/batchnorm/mul/ReadVariableOp¢'batch_normalization_319/AssignMovingAvg¢6batch_normalization_319/AssignMovingAvg/ReadVariableOp¢)batch_normalization_319/AssignMovingAvg_1¢8batch_normalization_319/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_319/batchnorm/ReadVariableOp¢4batch_normalization_319/batchnorm/mul/ReadVariableOp¢ dense_318/BiasAdd/ReadVariableOp¢dense_318/MatMul/ReadVariableOp¢ dense_319/BiasAdd/ReadVariableOp¢dense_319/MatMul/ReadVariableOp¢"input_layer/BiasAdd/ReadVariableOp¢!input_layer/MatMul/ReadVariableOp³
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource* 
_output_shapes
:
°*
dtype02#
!input_layer/MatMul/ReadVariableOp
input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
input_layer/MatMul±
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02$
"input_layer/BiasAdd/ReadVariableOp²
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
input_layer/BiasAdd}
input_layer/ReluReluinput_layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
input_layer/Relu{
dropout_318/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Û¹?2
dropout_318/dropout/Const°
dropout_318/dropout/MulMulinput_layer/Relu:activations:0"dropout_318/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout_318/dropout/Mul
dropout_318/dropout/ShapeShapeinput_layer/Relu:activations:0*
T0*
_output_shapes
:2
dropout_318/dropout/Shapeê
0dropout_318/dropout/random_uniform/RandomUniformRandomUniform"dropout_318/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*
dtype0*
seed2ÿÿÿÿ22
0dropout_318/dropout/random_uniform/RandomUniform
"dropout_318/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *R¸>2$
"dropout_318/dropout/GreaterEqual/yï
 dropout_318/dropout/GreaterEqualGreaterEqual9dropout_318/dropout/random_uniform/RandomUniform:output:0+dropout_318/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2"
 dropout_318/dropout/GreaterEqual¤
dropout_318/dropout/CastCast$dropout_318/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout_318/dropout/Cast«
dropout_318/dropout/Mul_1Muldropout_318/dropout/Mul:z:0dropout_318/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout_318/dropout/Mul_1º
6batch_normalization_318/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_318/moments/mean/reduction_indicesï
$batch_normalization_318/moments/meanMeandropout_318/dropout/Mul_1:z:0?batch_normalization_318/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	°*
	keep_dims(2&
$batch_normalization_318/moments/meanÅ
,batch_normalization_318/moments/StopGradientStopGradient-batch_normalization_318/moments/mean:output:0*
T0*
_output_shapes
:	°2.
,batch_normalization_318/moments/StopGradient
1batch_normalization_318/moments/SquaredDifferenceSquaredDifferencedropout_318/dropout/Mul_1:z:05batch_normalization_318/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°23
1batch_normalization_318/moments/SquaredDifferenceÂ
:batch_normalization_318/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_318/moments/variance/reduction_indices
(batch_normalization_318/moments/varianceMean5batch_normalization_318/moments/SquaredDifference:z:0Cbatch_normalization_318/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	°*
	keep_dims(2*
(batch_normalization_318/moments/varianceÉ
'batch_normalization_318/moments/SqueezeSqueeze-batch_normalization_318/moments/mean:output:0*
T0*
_output_shapes	
:°*
squeeze_dims
 2)
'batch_normalization_318/moments/SqueezeÑ
)batch_normalization_318/moments/Squeeze_1Squeeze1batch_normalization_318/moments/variance:output:0*
T0*
_output_shapes	
:°*
squeeze_dims
 2+
)batch_normalization_318/moments/Squeeze_1£
-batch_normalization_318/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_318/AssignMovingAvg/decayí
6batch_normalization_318/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_318_assignmovingavg_readvariableop_resource*
_output_shapes	
:°*
dtype028
6batch_normalization_318/AssignMovingAvg/ReadVariableOpù
+batch_normalization_318/AssignMovingAvg/subSub>batch_normalization_318/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_318/moments/Squeeze:output:0*
T0*
_output_shapes	
:°2-
+batch_normalization_318/AssignMovingAvg/subð
+batch_normalization_318/AssignMovingAvg/mulMul/batch_normalization_318/AssignMovingAvg/sub:z:06batch_normalization_318/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:°2-
+batch_normalization_318/AssignMovingAvg/mul·
'batch_normalization_318/AssignMovingAvgAssignSubVariableOp?batch_normalization_318_assignmovingavg_readvariableop_resource/batch_normalization_318/AssignMovingAvg/mul:z:07^batch_normalization_318/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_318/AssignMovingAvg§
/batch_normalization_318/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/batch_normalization_318/AssignMovingAvg_1/decayó
8batch_normalization_318/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_318_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:°*
dtype02:
8batch_normalization_318/AssignMovingAvg_1/ReadVariableOp
-batch_normalization_318/AssignMovingAvg_1/subSub@batch_normalization_318/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_318/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:°2/
-batch_normalization_318/AssignMovingAvg_1/subø
-batch_normalization_318/AssignMovingAvg_1/mulMul1batch_normalization_318/AssignMovingAvg_1/sub:z:08batch_normalization_318/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:°2/
-batch_normalization_318/AssignMovingAvg_1/mulÁ
)batch_normalization_318/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_318_assignmovingavg_1_readvariableop_resource1batch_normalization_318/AssignMovingAvg_1/mul:z:09^batch_normalization_318/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_318/AssignMovingAvg_1
'batch_normalization_318/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2)
'batch_normalization_318/batchnorm/add/yã
%batch_normalization_318/batchnorm/addAddV22batch_normalization_318/moments/Squeeze_1:output:00batch_normalization_318/batchnorm/add/y:output:0*
T0*
_output_shapes	
:°2'
%batch_normalization_318/batchnorm/add¬
'batch_normalization_318/batchnorm/RsqrtRsqrt)batch_normalization_318/batchnorm/add:z:0*
T0*
_output_shapes	
:°2)
'batch_normalization_318/batchnorm/Rsqrtç
4batch_normalization_318/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_318_batchnorm_mul_readvariableop_resource*
_output_shapes	
:°*
dtype026
4batch_normalization_318/batchnorm/mul/ReadVariableOpæ
%batch_normalization_318/batchnorm/mulMul+batch_normalization_318/batchnorm/Rsqrt:y:0<batch_normalization_318/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:°2'
%batch_normalization_318/batchnorm/mulÖ
'batch_normalization_318/batchnorm/mul_1Muldropout_318/dropout/Mul_1:z:0)batch_normalization_318/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2)
'batch_normalization_318/batchnorm/mul_1Ü
'batch_normalization_318/batchnorm/mul_2Mul0batch_normalization_318/moments/Squeeze:output:0)batch_normalization_318/batchnorm/mul:z:0*
T0*
_output_shapes	
:°2)
'batch_normalization_318/batchnorm/mul_2Û
0batch_normalization_318/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_318_batchnorm_readvariableop_resource*
_output_shapes	
:°*
dtype022
0batch_normalization_318/batchnorm/ReadVariableOpâ
%batch_normalization_318/batchnorm/subSub8batch_normalization_318/batchnorm/ReadVariableOp:value:0+batch_normalization_318/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:°2'
%batch_normalization_318/batchnorm/subæ
'batch_normalization_318/batchnorm/add_1AddV2+batch_normalization_318/batchnorm/mul_1:z:0)batch_normalization_318/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2)
'batch_normalization_318/batchnorm/add_1­
dense_318/MatMul/ReadVariableOpReadVariableOp(dense_318_matmul_readvariableop_resource* 
_output_shapes
:
°·*
dtype02!
dense_318/MatMul/ReadVariableOp·
dense_318/MatMulMatMul+batch_normalization_318/batchnorm/add_1:z:0'dense_318/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dense_318/MatMul«
 dense_318/BiasAdd/ReadVariableOpReadVariableOp)dense_318_biasadd_readvariableop_resource*
_output_shapes	
:·*
dtype02"
 dense_318/BiasAdd/ReadVariableOpª
dense_318/BiasAddBiasAdddense_318/MatMul:product:0(dense_318/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dense_318/BiasAddw
dense_318/ReluReludense_318/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dense_318/Relu{
dropout_319/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ú?2
dropout_319/dropout/Const®
dropout_319/dropout/MulMuldense_318/Relu:activations:0"dropout_319/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout_319/dropout/Mul
dropout_319/dropout/ShapeShapedense_318/Relu:activations:0*
T0*
_output_shapes
:2
dropout_319/dropout/Shapeæ
0dropout_319/dropout/random_uniform/RandomUniformRandomUniform"dropout_319/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*
dtype0*
seed222
0dropout_319/dropout/random_uniform/RandomUniform
"dropout_319/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ8>2$
"dropout_319/dropout/GreaterEqual/yï
 dropout_319/dropout/GreaterEqualGreaterEqual9dropout_319/dropout/random_uniform/RandomUniform:output:0+dropout_319/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2"
 dropout_319/dropout/GreaterEqual¤
dropout_319/dropout/CastCast$dropout_319/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout_319/dropout/Cast«
dropout_319/dropout/Mul_1Muldropout_319/dropout/Mul:z:0dropout_319/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout_319/dropout/Mul_1º
6batch_normalization_319/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_319/moments/mean/reduction_indicesï
$batch_normalization_319/moments/meanMeandropout_319/dropout/Mul_1:z:0?batch_normalization_319/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	·*
	keep_dims(2&
$batch_normalization_319/moments/meanÅ
,batch_normalization_319/moments/StopGradientStopGradient-batch_normalization_319/moments/mean:output:0*
T0*
_output_shapes
:	·2.
,batch_normalization_319/moments/StopGradient
1batch_normalization_319/moments/SquaredDifferenceSquaredDifferencedropout_319/dropout/Mul_1:z:05batch_normalization_319/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·23
1batch_normalization_319/moments/SquaredDifferenceÂ
:batch_normalization_319/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_319/moments/variance/reduction_indices
(batch_normalization_319/moments/varianceMean5batch_normalization_319/moments/SquaredDifference:z:0Cbatch_normalization_319/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	·*
	keep_dims(2*
(batch_normalization_319/moments/varianceÉ
'batch_normalization_319/moments/SqueezeSqueeze-batch_normalization_319/moments/mean:output:0*
T0*
_output_shapes	
:·*
squeeze_dims
 2)
'batch_normalization_319/moments/SqueezeÑ
)batch_normalization_319/moments/Squeeze_1Squeeze1batch_normalization_319/moments/variance:output:0*
T0*
_output_shapes	
:·*
squeeze_dims
 2+
)batch_normalization_319/moments/Squeeze_1£
-batch_normalization_319/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_319/AssignMovingAvg/decayí
6batch_normalization_319/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_319_assignmovingavg_readvariableop_resource*
_output_shapes	
:·*
dtype028
6batch_normalization_319/AssignMovingAvg/ReadVariableOpù
+batch_normalization_319/AssignMovingAvg/subSub>batch_normalization_319/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_319/moments/Squeeze:output:0*
T0*
_output_shapes	
:·2-
+batch_normalization_319/AssignMovingAvg/subð
+batch_normalization_319/AssignMovingAvg/mulMul/batch_normalization_319/AssignMovingAvg/sub:z:06batch_normalization_319/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:·2-
+batch_normalization_319/AssignMovingAvg/mul·
'batch_normalization_319/AssignMovingAvgAssignSubVariableOp?batch_normalization_319_assignmovingavg_readvariableop_resource/batch_normalization_319/AssignMovingAvg/mul:z:07^batch_normalization_319/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_319/AssignMovingAvg§
/batch_normalization_319/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/batch_normalization_319/AssignMovingAvg_1/decayó
8batch_normalization_319/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_319_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:·*
dtype02:
8batch_normalization_319/AssignMovingAvg_1/ReadVariableOp
-batch_normalization_319/AssignMovingAvg_1/subSub@batch_normalization_319/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_319/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:·2/
-batch_normalization_319/AssignMovingAvg_1/subø
-batch_normalization_319/AssignMovingAvg_1/mulMul1batch_normalization_319/AssignMovingAvg_1/sub:z:08batch_normalization_319/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:·2/
-batch_normalization_319/AssignMovingAvg_1/mulÁ
)batch_normalization_319/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_319_assignmovingavg_1_readvariableop_resource1batch_normalization_319/AssignMovingAvg_1/mul:z:09^batch_normalization_319/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_319/AssignMovingAvg_1
'batch_normalization_319/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2)
'batch_normalization_319/batchnorm/add/yã
%batch_normalization_319/batchnorm/addAddV22batch_normalization_319/moments/Squeeze_1:output:00batch_normalization_319/batchnorm/add/y:output:0*
T0*
_output_shapes	
:·2'
%batch_normalization_319/batchnorm/add¬
'batch_normalization_319/batchnorm/RsqrtRsqrt)batch_normalization_319/batchnorm/add:z:0*
T0*
_output_shapes	
:·2)
'batch_normalization_319/batchnorm/Rsqrtç
4batch_normalization_319/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_319_batchnorm_mul_readvariableop_resource*
_output_shapes	
:·*
dtype026
4batch_normalization_319/batchnorm/mul/ReadVariableOpæ
%batch_normalization_319/batchnorm/mulMul+batch_normalization_319/batchnorm/Rsqrt:y:0<batch_normalization_319/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:·2'
%batch_normalization_319/batchnorm/mulÖ
'batch_normalization_319/batchnorm/mul_1Muldropout_319/dropout/Mul_1:z:0)batch_normalization_319/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2)
'batch_normalization_319/batchnorm/mul_1Ü
'batch_normalization_319/batchnorm/mul_2Mul0batch_normalization_319/moments/Squeeze:output:0)batch_normalization_319/batchnorm/mul:z:0*
T0*
_output_shapes	
:·2)
'batch_normalization_319/batchnorm/mul_2Û
0batch_normalization_319/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_319_batchnorm_readvariableop_resource*
_output_shapes	
:·*
dtype022
0batch_normalization_319/batchnorm/ReadVariableOpâ
%batch_normalization_319/batchnorm/subSub8batch_normalization_319/batchnorm/ReadVariableOp:value:0+batch_normalization_319/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:·2'
%batch_normalization_319/batchnorm/subæ
'batch_normalization_319/batchnorm/add_1AddV2+batch_normalization_319/batchnorm/mul_1:z:0)batch_normalization_319/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2)
'batch_normalization_319/batchnorm/add_1¬
dense_319/MatMul/ReadVariableOpReadVariableOp(dense_319_matmul_readvariableop_resource*
_output_shapes
:	·*
dtype02!
dense_319/MatMul/ReadVariableOp¶
dense_319/MatMulMatMul+batch_normalization_319/batchnorm/add_1:z:0'dense_319/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_319/MatMulª
 dense_319/BiasAdd/ReadVariableOpReadVariableOp)dense_319_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_319/BiasAdd/ReadVariableOp©
dense_319/BiasAddBiasAdddense_319/MatMul:product:0(dense_319/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_319/BiasAdd
dense_319/SoftmaxSoftmaxdense_319/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_319/Softmaxª
IdentityIdentitydense_319/Softmax:softmax:0(^batch_normalization_318/AssignMovingAvg7^batch_normalization_318/AssignMovingAvg/ReadVariableOp*^batch_normalization_318/AssignMovingAvg_19^batch_normalization_318/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_318/batchnorm/ReadVariableOp5^batch_normalization_318/batchnorm/mul/ReadVariableOp(^batch_normalization_319/AssignMovingAvg7^batch_normalization_319/AssignMovingAvg/ReadVariableOp*^batch_normalization_319/AssignMovingAvg_19^batch_normalization_319/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_319/batchnorm/ReadVariableOp5^batch_normalization_319/batchnorm/mul/ReadVariableOp!^dense_318/BiasAdd/ReadVariableOp ^dense_318/MatMul/ReadVariableOp!^dense_319/BiasAdd/ReadVariableOp ^dense_319/MatMul/ReadVariableOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2R
'batch_normalization_318/AssignMovingAvg'batch_normalization_318/AssignMovingAvg2p
6batch_normalization_318/AssignMovingAvg/ReadVariableOp6batch_normalization_318/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_318/AssignMovingAvg_1)batch_normalization_318/AssignMovingAvg_12t
8batch_normalization_318/AssignMovingAvg_1/ReadVariableOp8batch_normalization_318/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_318/batchnorm/ReadVariableOp0batch_normalization_318/batchnorm/ReadVariableOp2l
4batch_normalization_318/batchnorm/mul/ReadVariableOp4batch_normalization_318/batchnorm/mul/ReadVariableOp2R
'batch_normalization_319/AssignMovingAvg'batch_normalization_319/AssignMovingAvg2p
6batch_normalization_319/AssignMovingAvg/ReadVariableOp6batch_normalization_319/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_319/AssignMovingAvg_1)batch_normalization_319/AssignMovingAvg_12t
8batch_normalization_319/AssignMovingAvg_1/ReadVariableOp8batch_normalization_319/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_319/batchnorm/ReadVariableOp0batch_normalization_319/batchnorm/ReadVariableOp2l
4batch_normalization_319/batchnorm/mul/ReadVariableOp4batch_normalization_319/batchnorm/mul/ReadVariableOp2D
 dense_318/BiasAdd/ReadVariableOp dense_318/BiasAdd/ReadVariableOp2B
dense_318/MatMul/ReadVariableOpdense_318/MatMul/ReadVariableOp2D
 dense_319/BiasAdd/ReadVariableOp dense_319/BiasAdd/ReadVariableOp2B
dense_319/MatMul/ReadVariableOpdense_319/MatMul/ReadVariableOp2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
·
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_8288431

inputs0
!batchnorm_readvariableop_resource:	°4
%batchnorm_mul_readvariableop_resource:	°2
#batchnorm_readvariableop_1_resource:	°2
#batchnorm_readvariableop_2_resource:	°
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:°*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:°2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:°2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:°*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:°2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:°*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:°2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:°*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:°2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
³
·
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_8287383

inputs0
!batchnorm_readvariableop_resource:	°4
%batchnorm_mul_readvariableop_resource:	°2
#batchnorm_readvariableop_1_resource:	°2
#batchnorm_readvariableop_2_resource:	°
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:°*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:°2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:°2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:°*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:°2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:°*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:°2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:°*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:°2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
³
·
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_8287545

inputs0
!batchnorm_readvariableop_resource:	·4
%batchnorm_mul_readvariableop_resource:	·2
#batchnorm_readvariableop_1_resource:	·2
#batchnorm_readvariableop_2_resource:	·
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:·*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:·2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:·2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:·*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:·2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:·*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:·2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:·*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:·2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs

Ø
9__inference_batch_normalization_319_layer_call_fn_8288618

inputs
unknown:	·
	unknown_0:	·
	unknown_1:	·
	unknown_2:	·
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_82876052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
È
g
H__inference_dropout_318_layer_call_and_return_conditional_losses_8288401

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Û¹?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÆ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*
dtype0*
seed2ÿÿÿÿ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *R¸>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
¼

ü
H__inference_input_layer_layer_call_and_return_conditional_losses_8287701

inputs2
matmul_readvariableop_resource:
°.
biasadd_readvariableop_resource:	°
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
ð
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288364

inputs
unknown:
°
	unknown_0:	°
	unknown_1:	°
	unknown_2:	°
	unknown_3:	°
	unknown_4:	°
	unknown_5:
°·
	unknown_6:	·
	unknown_7:	·
	unknown_8:	·
	unknown_9:	·

unknown_10:	·

unknown_11:	·

unknown_12:
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *a
f\RZ
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_82879552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

-__inference_input_layer_layer_call_fn_8288384

inputs
unknown:
°
	unknown_0:	°
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_82877012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
*
ñ
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_8288592

inputs6
'assignmovingavg_readvariableop_resource:	·8
)assignmovingavg_1_readvariableop_resource:	·4
%batchnorm_mul_readvariableop_resource:	·0
!batchnorm_readvariableop_resource:	·
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	·*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	·2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	·*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:·*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:·*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:·*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:·2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:·2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:·*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:·2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:·2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:·2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:·2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:·*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:·2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:·2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:·*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:·2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs

Ø
9__inference_batch_normalization_319_layer_call_fn_8288605

inputs
unknown:	·
	unknown_0:	·
	unknown_1:	·
	unknown_2:	·
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_82875452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs

ã
%__inference_signature_wrapper_8288138
input_layer_input
unknown:
°
	unknown_0:	°
	unknown_1:	°
	unknown_2:	°
	unknown_3:	°
	unknown_4:	°
	unknown_5:
°·
	unknown_6:	·
	unknown_7:	·
	unknown_8:	·
	unknown_9:	·

unknown_10:	·

unknown_11:	·

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__wrapped_model_82873592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameinput_layer_input
ý»
ø
#__inference__traced_restore_8288929
file_prefix7
#assignvariableop_input_layer_kernel:
°2
#assignvariableop_1_input_layer_bias:	°?
0assignvariableop_2_batch_normalization_318_gamma:	°>
/assignvariableop_3_batch_normalization_318_beta:	°E
6assignvariableop_4_batch_normalization_318_moving_mean:	°I
:assignvariableop_5_batch_normalization_318_moving_variance:	°7
#assignvariableop_6_dense_318_kernel:
°·0
!assignvariableop_7_dense_318_bias:	·?
0assignvariableop_8_batch_normalization_319_gamma:	·>
/assignvariableop_9_batch_normalization_319_beta:	·F
7assignvariableop_10_batch_normalization_319_moving_mean:	·J
;assignvariableop_11_batch_normalization_319_moving_variance:	·7
$assignvariableop_12_dense_319_kernel:	·0
"assignvariableop_13_dense_319_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: A
-assignvariableop_23_adam_input_layer_kernel_m:
°:
+assignvariableop_24_adam_input_layer_bias_m:	°G
8assignvariableop_25_adam_batch_normalization_318_gamma_m:	°F
7assignvariableop_26_adam_batch_normalization_318_beta_m:	°?
+assignvariableop_27_adam_dense_318_kernel_m:
°·8
)assignvariableop_28_adam_dense_318_bias_m:	·G
8assignvariableop_29_adam_batch_normalization_319_gamma_m:	·F
7assignvariableop_30_adam_batch_normalization_319_beta_m:	·>
+assignvariableop_31_adam_dense_319_kernel_m:	·7
)assignvariableop_32_adam_dense_319_bias_m:A
-assignvariableop_33_adam_input_layer_kernel_v:
°:
+assignvariableop_34_adam_input_layer_bias_v:	°G
8assignvariableop_35_adam_batch_normalization_318_gamma_v:	°F
7assignvariableop_36_adam_batch_normalization_318_beta_v:	°?
+assignvariableop_37_adam_dense_318_kernel_v:
°·8
)assignvariableop_38_adam_dense_318_bias_v:	·G
8assignvariableop_39_adam_batch_normalization_319_gamma_v:	·F
7assignvariableop_40_adam_batch_normalization_319_beta_v:	·>
+assignvariableop_41_adam_dense_319_kernel_v:	·7
)assignvariableop_42_adam_dense_319_bias_v:
identity_44¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ü
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*
valueþBû,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesæ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¢
AssignVariableOpAssignVariableOp#assignvariableop_input_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¨
AssignVariableOp_1AssignVariableOp#assignvariableop_1_input_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2µ
AssignVariableOp_2AssignVariableOp0assignvariableop_2_batch_normalization_318_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3´
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_318_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4»
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_318_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¿
AssignVariableOp_5AssignVariableOp:assignvariableop_5_batch_normalization_318_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_318_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_318_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8µ
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batch_normalization_319_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9´
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_319_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¿
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_319_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ã
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batch_normalization_319_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¬
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_319_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ª
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_319_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14¥
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15§
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16§
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¦
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¡
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¡
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22£
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23µ
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_input_layer_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24³
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_input_layer_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25À
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_batch_normalization_318_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¿
AssignVariableOp_26AssignVariableOp7assignvariableop_26_adam_batch_normalization_318_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_318_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_318_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29À
AssignVariableOp_29AssignVariableOp8assignvariableop_29_adam_batch_normalization_319_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¿
AssignVariableOp_30AssignVariableOp7assignvariableop_30_adam_batch_normalization_319_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_319_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_319_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33µ
AssignVariableOp_33AssignVariableOp-assignvariableop_33_adam_input_layer_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34³
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_input_layer_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35À
AssignVariableOp_35AssignVariableOp8assignvariableop_35_adam_batch_normalization_318_gamma_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¿
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adam_batch_normalization_318_beta_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_318_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_318_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39À
AssignVariableOp_39AssignVariableOp8assignvariableop_39_adam_batch_normalization_319_gamma_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¿
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_batch_normalization_319_beta_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41³
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_319_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42±
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_319_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
å\
Û
 __inference__traced_save_8288790
file_prefix1
-savev2_input_layer_kernel_read_readvariableop/
+savev2_input_layer_bias_read_readvariableop<
8savev2_batch_normalization_318_gamma_read_readvariableop;
7savev2_batch_normalization_318_beta_read_readvariableopB
>savev2_batch_normalization_318_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_318_moving_variance_read_readvariableop/
+savev2_dense_318_kernel_read_readvariableop-
)savev2_dense_318_bias_read_readvariableop<
8savev2_batch_normalization_319_gamma_read_readvariableop;
7savev2_batch_normalization_319_beta_read_readvariableopB
>savev2_batch_normalization_319_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_319_moving_variance_read_readvariableop/
+savev2_dense_319_kernel_read_readvariableop-
)savev2_dense_319_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop8
4savev2_adam_input_layer_kernel_m_read_readvariableop6
2savev2_adam_input_layer_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_318_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_318_beta_m_read_readvariableop6
2savev2_adam_dense_318_kernel_m_read_readvariableop4
0savev2_adam_dense_318_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_319_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_319_beta_m_read_readvariableop6
2savev2_adam_dense_319_kernel_m_read_readvariableop4
0savev2_adam_dense_319_bias_m_read_readvariableop8
4savev2_adam_input_layer_kernel_v_read_readvariableop6
2savev2_adam_input_layer_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_318_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_318_beta_v_read_readvariableop6
2savev2_adam_dense_318_kernel_v_read_readvariableop4
0savev2_adam_dense_318_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_319_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_319_beta_v_read_readvariableop6
2savev2_adam_dense_319_kernel_v_read_readvariableop4
0savev2_adam_dense_319_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameö
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*
valueþBû,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesà
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices§
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_input_layer_kernel_read_readvariableop+savev2_input_layer_bias_read_readvariableop8savev2_batch_normalization_318_gamma_read_readvariableop7savev2_batch_normalization_318_beta_read_readvariableop>savev2_batch_normalization_318_moving_mean_read_readvariableopBsavev2_batch_normalization_318_moving_variance_read_readvariableop+savev2_dense_318_kernel_read_readvariableop)savev2_dense_318_bias_read_readvariableop8savev2_batch_normalization_319_gamma_read_readvariableop7savev2_batch_normalization_319_beta_read_readvariableop>savev2_batch_normalization_319_moving_mean_read_readvariableopBsavev2_batch_normalization_319_moving_variance_read_readvariableop+savev2_dense_319_kernel_read_readvariableop)savev2_dense_319_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop4savev2_adam_input_layer_kernel_m_read_readvariableop2savev2_adam_input_layer_bias_m_read_readvariableop?savev2_adam_batch_normalization_318_gamma_m_read_readvariableop>savev2_adam_batch_normalization_318_beta_m_read_readvariableop2savev2_adam_dense_318_kernel_m_read_readvariableop0savev2_adam_dense_318_bias_m_read_readvariableop?savev2_adam_batch_normalization_319_gamma_m_read_readvariableop>savev2_adam_batch_normalization_319_beta_m_read_readvariableop2savev2_adam_dense_319_kernel_m_read_readvariableop0savev2_adam_dense_319_bias_m_read_readvariableop4savev2_adam_input_layer_kernel_v_read_readvariableop2savev2_adam_input_layer_bias_v_read_readvariableop?savev2_adam_batch_normalization_318_gamma_v_read_readvariableop>savev2_adam_batch_normalization_318_beta_v_read_readvariableop2savev2_adam_dense_318_kernel_v_read_readvariableop0savev2_adam_dense_318_bias_v_read_readvariableop?savev2_adam_batch_normalization_319_gamma_v_read_readvariableop>savev2_adam_batch_normalization_319_beta_v_read_readvariableop2savev2_adam_dense_319_kernel_v_read_readvariableop0savev2_adam_dense_319_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*À
_input_shapes®
«: :
°:°:°:°:°:°:
°·:·:·:·:·:·:	·:: : : : : : : : : :
°:°:°:°:
°·:·:·:·:	·::
°:°:°:°:
°·:·:·:·:	·:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
°:!

_output_shapes	
:°:!

_output_shapes	
:°:!

_output_shapes	
:°:!

_output_shapes	
:°:!

_output_shapes	
:°:&"
 
_output_shapes
:
°·:!

_output_shapes	
:·:!	

_output_shapes	
:·:!


_output_shapes	
:·:!

_output_shapes	
:·:!

_output_shapes	
:·:%!

_output_shapes
:	·: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
°:!

_output_shapes	
:°:!

_output_shapes	
:°:!

_output_shapes	
:°:&"
 
_output_shapes
:
°·:!

_output_shapes	
:·:!

_output_shapes	
:·:!

_output_shapes	
:·:% !

_output_shapes
:	·: !

_output_shapes
::&""
 
_output_shapes
:
°:!#

_output_shapes	
:°:!$

_output_shapes	
:°:!%

_output_shapes	
:°:&&"
 
_output_shapes
:
°·:!'

_output_shapes	
:·:!(

_output_shapes	
:·:!)

_output_shapes	
:·:%*!

_output_shapes
:	·: +

_output_shapes
::,

_output_shapes
: 
º

ú
F__inference_dense_318_layer_call_and_return_conditional_losses_8288502

inputs2
matmul_readvariableop_resource:
°·.
biasadd_readvariableop_resource:	·
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
°·*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:·*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
Å,
ý
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288097
input_layer_input'
input_layer_8288061:
°"
input_layer_8288063:	°.
batch_normalization_318_8288067:	°.
batch_normalization_318_8288069:	°.
batch_normalization_318_8288071:	°.
batch_normalization_318_8288073:	°%
dense_318_8288076:
°· 
dense_318_8288078:	·.
batch_normalization_319_8288082:	·.
batch_normalization_319_8288084:	·.
batch_normalization_319_8288086:	·.
batch_normalization_319_8288088:	·$
dense_319_8288091:	·
dense_319_8288093:
identity¢/batch_normalization_318/StatefulPartitionedCall¢/batch_normalization_319/StatefulPartitionedCall¢!dense_318/StatefulPartitionedCall¢!dense_319/StatefulPartitionedCall¢#dropout_318/StatefulPartitionedCall¢#dropout_319/StatefulPartitionedCall¢#input_layer/StatefulPartitionedCall·
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputinput_layer_8288061input_layer_8288063*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_82877012%
#input_layer/StatefulPartitionedCall¢
#dropout_318/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_318_layer_call_and_return_conditional_losses_82878682%
#dropout_318/StatefulPartitionedCallÒ
/batch_normalization_318/StatefulPartitionedCallStatefulPartitionedCall,dropout_318/StatefulPartitionedCall:output:0batch_normalization_318_8288067batch_normalization_318_8288069batch_normalization_318_8288071batch_normalization_318_8288073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_828744321
/batch_normalization_318/StatefulPartitionedCallÔ
!dense_318/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_318/StatefulPartitionedCall:output:0dense_318_8288076dense_318_8288078*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_82877342#
!dense_318/StatefulPartitionedCallÆ
#dropout_319/StatefulPartitionedCallStatefulPartitionedCall*dense_318/StatefulPartitionedCall:output:0$^dropout_318/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_319_layer_call_and_return_conditional_losses_82878352%
#dropout_319/StatefulPartitionedCallÒ
/batch_normalization_319/StatefulPartitionedCallStatefulPartitionedCall,dropout_319/StatefulPartitionedCall:output:0batch_normalization_319_8288082batch_normalization_319_8288084batch_normalization_319_8288086batch_normalization_319_8288088*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_828760521
/batch_normalization_319/StatefulPartitionedCallÓ
!dense_319/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_319/StatefulPartitionedCall:output:0dense_319_8288091dense_319_8288093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_82877672#
!dense_319/StatefulPartitionedCall
IdentityIdentity*dense_319/StatefulPartitionedCall:output:00^batch_normalization_318/StatefulPartitionedCall0^batch_normalization_319/StatefulPartitionedCall"^dense_318/StatefulPartitionedCall"^dense_319/StatefulPartitionedCall$^dropout_318/StatefulPartitionedCall$^dropout_319/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2b
/batch_normalization_318/StatefulPartitionedCall/batch_normalization_318/StatefulPartitionedCall2b
/batch_normalization_319/StatefulPartitionedCall/batch_normalization_319/StatefulPartitionedCall2F
!dense_318/StatefulPartitionedCall!dense_318/StatefulPartitionedCall2F
!dense_319/StatefulPartitionedCall!dense_319/StatefulPartitionedCall2J
#dropout_318/StatefulPartitionedCall#dropout_318/StatefulPartitionedCall2J
#dropout_319/StatefulPartitionedCall#dropout_319/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameinput_layer_input
*
ñ
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_8287605

inputs6
'assignmovingavg_readvariableop_resource:	·8
)assignmovingavg_1_readvariableop_resource:	·4
%batchnorm_mul_readvariableop_resource:	·0
!batchnorm_readvariableop_resource:	·
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	·*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	·2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	·*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:·*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:·*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:·*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:·2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:·2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:·*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:·2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:·2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:·2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:·2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:·*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:·2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:·2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:·*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:·2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
ù
f
H__inference_dropout_318_layer_call_and_return_conditional_losses_8288389

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs

Ø
9__inference_batch_normalization_318_layer_call_fn_8288491

inputs
unknown:	°
	unknown_0:	°
	unknown_1:	°
	unknown_2:	°
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_82874432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
ä
û
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288019
input_layer_input
unknown:
°
	unknown_0:	°
	unknown_1:	°
	unknown_2:	°
	unknown_3:	°
	unknown_4:	°
	unknown_5:
°·
	unknown_6:	·
	unknown_7:	·
	unknown_8:	·
	unknown_9:	·

unknown_10:	·

unknown_11:	·

unknown_12:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *a
f\RZ
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_82879552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameinput_layer_input
¼

ü
H__inference_input_layer_layer_call_and_return_conditional_losses_8288375

inputs2
matmul_readvariableop_resource:
°.
biasadd_readvariableop_resource:	°
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ]
¼
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288197

inputs>
*input_layer_matmul_readvariableop_resource:
°:
+input_layer_biasadd_readvariableop_resource:	°H
9batch_normalization_318_batchnorm_readvariableop_resource:	°L
=batch_normalization_318_batchnorm_mul_readvariableop_resource:	°J
;batch_normalization_318_batchnorm_readvariableop_1_resource:	°J
;batch_normalization_318_batchnorm_readvariableop_2_resource:	°<
(dense_318_matmul_readvariableop_resource:
°·8
)dense_318_biasadd_readvariableop_resource:	·H
9batch_normalization_319_batchnorm_readvariableop_resource:	·L
=batch_normalization_319_batchnorm_mul_readvariableop_resource:	·J
;batch_normalization_319_batchnorm_readvariableop_1_resource:	·J
;batch_normalization_319_batchnorm_readvariableop_2_resource:	·;
(dense_319_matmul_readvariableop_resource:	·7
)dense_319_biasadd_readvariableop_resource:
identity¢0batch_normalization_318/batchnorm/ReadVariableOp¢2batch_normalization_318/batchnorm/ReadVariableOp_1¢2batch_normalization_318/batchnorm/ReadVariableOp_2¢4batch_normalization_318/batchnorm/mul/ReadVariableOp¢0batch_normalization_319/batchnorm/ReadVariableOp¢2batch_normalization_319/batchnorm/ReadVariableOp_1¢2batch_normalization_319/batchnorm/ReadVariableOp_2¢4batch_normalization_319/batchnorm/mul/ReadVariableOp¢ dense_318/BiasAdd/ReadVariableOp¢dense_318/MatMul/ReadVariableOp¢ dense_319/BiasAdd/ReadVariableOp¢dense_319/MatMul/ReadVariableOp¢"input_layer/BiasAdd/ReadVariableOp¢!input_layer/MatMul/ReadVariableOp³
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource* 
_output_shapes
:
°*
dtype02#
!input_layer/MatMul/ReadVariableOp
input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
input_layer/MatMul±
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02$
"input_layer/BiasAdd/ReadVariableOp²
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
input_layer/BiasAdd}
input_layer/ReluReluinput_layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
input_layer/Relu
dropout_318/IdentityIdentityinput_layer/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout_318/IdentityÛ
0batch_normalization_318/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_318_batchnorm_readvariableop_resource*
_output_shapes	
:°*
dtype022
0batch_normalization_318/batchnorm/ReadVariableOp
'batch_normalization_318/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2)
'batch_normalization_318/batchnorm/add/yé
%batch_normalization_318/batchnorm/addAddV28batch_normalization_318/batchnorm/ReadVariableOp:value:00batch_normalization_318/batchnorm/add/y:output:0*
T0*
_output_shapes	
:°2'
%batch_normalization_318/batchnorm/add¬
'batch_normalization_318/batchnorm/RsqrtRsqrt)batch_normalization_318/batchnorm/add:z:0*
T0*
_output_shapes	
:°2)
'batch_normalization_318/batchnorm/Rsqrtç
4batch_normalization_318/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_318_batchnorm_mul_readvariableop_resource*
_output_shapes	
:°*
dtype026
4batch_normalization_318/batchnorm/mul/ReadVariableOpæ
%batch_normalization_318/batchnorm/mulMul+batch_normalization_318/batchnorm/Rsqrt:y:0<batch_normalization_318/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:°2'
%batch_normalization_318/batchnorm/mulÖ
'batch_normalization_318/batchnorm/mul_1Muldropout_318/Identity:output:0)batch_normalization_318/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2)
'batch_normalization_318/batchnorm/mul_1á
2batch_normalization_318/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_318_batchnorm_readvariableop_1_resource*
_output_shapes	
:°*
dtype024
2batch_normalization_318/batchnorm/ReadVariableOp_1æ
'batch_normalization_318/batchnorm/mul_2Mul:batch_normalization_318/batchnorm/ReadVariableOp_1:value:0)batch_normalization_318/batchnorm/mul:z:0*
T0*
_output_shapes	
:°2)
'batch_normalization_318/batchnorm/mul_2á
2batch_normalization_318/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_318_batchnorm_readvariableop_2_resource*
_output_shapes	
:°*
dtype024
2batch_normalization_318/batchnorm/ReadVariableOp_2ä
%batch_normalization_318/batchnorm/subSub:batch_normalization_318/batchnorm/ReadVariableOp_2:value:0+batch_normalization_318/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:°2'
%batch_normalization_318/batchnorm/subæ
'batch_normalization_318/batchnorm/add_1AddV2+batch_normalization_318/batchnorm/mul_1:z:0)batch_normalization_318/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2)
'batch_normalization_318/batchnorm/add_1­
dense_318/MatMul/ReadVariableOpReadVariableOp(dense_318_matmul_readvariableop_resource* 
_output_shapes
:
°·*
dtype02!
dense_318/MatMul/ReadVariableOp·
dense_318/MatMulMatMul+batch_normalization_318/batchnorm/add_1:z:0'dense_318/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dense_318/MatMul«
 dense_318/BiasAdd/ReadVariableOpReadVariableOp)dense_318_biasadd_readvariableop_resource*
_output_shapes	
:·*
dtype02"
 dense_318/BiasAdd/ReadVariableOpª
dense_318/BiasAddBiasAdddense_318/MatMul:product:0(dense_318/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dense_318/BiasAddw
dense_318/ReluReludense_318/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dense_318/Relu
dropout_319/IdentityIdentitydense_318/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout_319/IdentityÛ
0batch_normalization_319/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_319_batchnorm_readvariableop_resource*
_output_shapes	
:·*
dtype022
0batch_normalization_319/batchnorm/ReadVariableOp
'batch_normalization_319/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2)
'batch_normalization_319/batchnorm/add/yé
%batch_normalization_319/batchnorm/addAddV28batch_normalization_319/batchnorm/ReadVariableOp:value:00batch_normalization_319/batchnorm/add/y:output:0*
T0*
_output_shapes	
:·2'
%batch_normalization_319/batchnorm/add¬
'batch_normalization_319/batchnorm/RsqrtRsqrt)batch_normalization_319/batchnorm/add:z:0*
T0*
_output_shapes	
:·2)
'batch_normalization_319/batchnorm/Rsqrtç
4batch_normalization_319/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_319_batchnorm_mul_readvariableop_resource*
_output_shapes	
:·*
dtype026
4batch_normalization_319/batchnorm/mul/ReadVariableOpæ
%batch_normalization_319/batchnorm/mulMul+batch_normalization_319/batchnorm/Rsqrt:y:0<batch_normalization_319/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:·2'
%batch_normalization_319/batchnorm/mulÖ
'batch_normalization_319/batchnorm/mul_1Muldropout_319/Identity:output:0)batch_normalization_319/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2)
'batch_normalization_319/batchnorm/mul_1á
2batch_normalization_319/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_319_batchnorm_readvariableop_1_resource*
_output_shapes	
:·*
dtype024
2batch_normalization_319/batchnorm/ReadVariableOp_1æ
'batch_normalization_319/batchnorm/mul_2Mul:batch_normalization_319/batchnorm/ReadVariableOp_1:value:0)batch_normalization_319/batchnorm/mul:z:0*
T0*
_output_shapes	
:·2)
'batch_normalization_319/batchnorm/mul_2á
2batch_normalization_319/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_319_batchnorm_readvariableop_2_resource*
_output_shapes	
:·*
dtype024
2batch_normalization_319/batchnorm/ReadVariableOp_2ä
%batch_normalization_319/batchnorm/subSub:batch_normalization_319/batchnorm/ReadVariableOp_2:value:0+batch_normalization_319/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:·2'
%batch_normalization_319/batchnorm/subæ
'batch_normalization_319/batchnorm/add_1AddV2+batch_normalization_319/batchnorm/mul_1:z:0)batch_normalization_319/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2)
'batch_normalization_319/batchnorm/add_1¬
dense_319/MatMul/ReadVariableOpReadVariableOp(dense_319_matmul_readvariableop_resource*
_output_shapes
:	·*
dtype02!
dense_319/MatMul/ReadVariableOp¶
dense_319/MatMulMatMul+batch_normalization_319/batchnorm/add_1:z:0'dense_319/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_319/MatMulª
 dense_319/BiasAdd/ReadVariableOpReadVariableOp)dense_319_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_319/BiasAdd/ReadVariableOp©
dense_319/BiasAddBiasAdddense_319/MatMul:product:0(dense_319/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_319/BiasAdd
dense_319/SoftmaxSoftmaxdense_319/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_319/Softmaxê
IdentityIdentitydense_319/Softmax:softmax:01^batch_normalization_318/batchnorm/ReadVariableOp3^batch_normalization_318/batchnorm/ReadVariableOp_13^batch_normalization_318/batchnorm/ReadVariableOp_25^batch_normalization_318/batchnorm/mul/ReadVariableOp1^batch_normalization_319/batchnorm/ReadVariableOp3^batch_normalization_319/batchnorm/ReadVariableOp_13^batch_normalization_319/batchnorm/ReadVariableOp_25^batch_normalization_319/batchnorm/mul/ReadVariableOp!^dense_318/BiasAdd/ReadVariableOp ^dense_318/MatMul/ReadVariableOp!^dense_319/BiasAdd/ReadVariableOp ^dense_319/MatMul/ReadVariableOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2d
0batch_normalization_318/batchnorm/ReadVariableOp0batch_normalization_318/batchnorm/ReadVariableOp2h
2batch_normalization_318/batchnorm/ReadVariableOp_12batch_normalization_318/batchnorm/ReadVariableOp_12h
2batch_normalization_318/batchnorm/ReadVariableOp_22batch_normalization_318/batchnorm/ReadVariableOp_22l
4batch_normalization_318/batchnorm/mul/ReadVariableOp4batch_normalization_318/batchnorm/mul/ReadVariableOp2d
0batch_normalization_319/batchnorm/ReadVariableOp0batch_normalization_319/batchnorm/ReadVariableOp2h
2batch_normalization_319/batchnorm/ReadVariableOp_12batch_normalization_319/batchnorm/ReadVariableOp_12h
2batch_normalization_319/batchnorm/ReadVariableOp_22batch_normalization_319/batchnorm/ReadVariableOp_22l
4batch_normalization_319/batchnorm/mul/ReadVariableOp4batch_normalization_319/batchnorm/mul/ReadVariableOp2D
 dense_318/BiasAdd/ReadVariableOp dense_318/BiasAdd/ReadVariableOp2B
dense_318/MatMul/ReadVariableOpdense_318/MatMul/ReadVariableOp2D
 dense_319/BiasAdd/ReadVariableOp dense_319/BiasAdd/ReadVariableOp2B
dense_319/MatMul/ReadVariableOpdense_319/MatMul/ReadVariableOp2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
I
-__inference_dropout_319_layer_call_fn_8288533

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_319_layer_call_and_return_conditional_losses_82877452
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
«

+__inference_dense_318_layer_call_fn_8288511

inputs
unknown:
°·
	unknown_0:	·
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_82877342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
º

ø
F__inference_dense_319_layer_call_and_return_conditional_losses_8288629

inputs1
matmul_readvariableop_resource:	·-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	·*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
§

+__inference_dense_319_layer_call_fn_8288638

inputs
unknown:	·
	unknown_0:
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_82877672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
³)
±
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288058
input_layer_input'
input_layer_8288022:
°"
input_layer_8288024:	°.
batch_normalization_318_8288028:	°.
batch_normalization_318_8288030:	°.
batch_normalization_318_8288032:	°.
batch_normalization_318_8288034:	°%
dense_318_8288037:
°· 
dense_318_8288039:	·.
batch_normalization_319_8288043:	·.
batch_normalization_319_8288045:	·.
batch_normalization_319_8288047:	·.
batch_normalization_319_8288049:	·$
dense_319_8288052:	·
dense_319_8288054:
identity¢/batch_normalization_318/StatefulPartitionedCall¢/batch_normalization_319/StatefulPartitionedCall¢!dense_318/StatefulPartitionedCall¢!dense_319/StatefulPartitionedCall¢#input_layer/StatefulPartitionedCall·
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputinput_layer_8288022input_layer_8288024*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_82877012%
#input_layer/StatefulPartitionedCall
dropout_318/PartitionedCallPartitionedCall,input_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_318_layer_call_and_return_conditional_losses_82877122
dropout_318/PartitionedCallÌ
/batch_normalization_318/StatefulPartitionedCallStatefulPartitionedCall$dropout_318/PartitionedCall:output:0batch_normalization_318_8288028batch_normalization_318_8288030batch_normalization_318_8288032batch_normalization_318_8288034*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_828738321
/batch_normalization_318/StatefulPartitionedCallÔ
!dense_318/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_318/StatefulPartitionedCall:output:0dense_318_8288037dense_318_8288039*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_82877342#
!dense_318/StatefulPartitionedCall
dropout_319/PartitionedCallPartitionedCall*dense_318/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_319_layer_call_and_return_conditional_losses_82877452
dropout_319/PartitionedCallÌ
/batch_normalization_319/StatefulPartitionedCallStatefulPartitionedCall$dropout_319/PartitionedCall:output:0batch_normalization_319_8288043batch_normalization_319_8288045batch_normalization_319_8288047batch_normalization_319_8288049*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_828754521
/batch_normalization_319/StatefulPartitionedCallÓ
!dense_319/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_319/StatefulPartitionedCall:output:0dense_319_8288052dense_319_8288054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_82877672#
!dense_319/StatefulPartitionedCallÐ
IdentityIdentity*dense_319/StatefulPartitionedCall:output:00^batch_normalization_318/StatefulPartitionedCall0^batch_normalization_319/StatefulPartitionedCall"^dense_318/StatefulPartitionedCall"^dense_319/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2b
/batch_normalization_318/StatefulPartitionedCall/batch_normalization_318/StatefulPartitionedCall2b
/batch_normalization_319/StatefulPartitionedCall/batch_normalization_319/StatefulPartitionedCall2F
!dense_318/StatefulPartitionedCall!dense_318/StatefulPartitionedCall2F
!dense_319/StatefulPartitionedCall!dense_319/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameinput_layer_input
ù
f
H__inference_dropout_318_layer_call_and_return_conditional_losses_8287712

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
*
ñ
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_8287443

inputs6
'assignmovingavg_readvariableop_resource:	°8
)assignmovingavg_1_readvariableop_resource:	°4
%batchnorm_mul_readvariableop_resource:	°0
!batchnorm_readvariableop_resource:	°
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	°*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	°2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	°*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:°*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:°*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:°*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:°2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:°2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:°*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:°2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:°2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:°2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:°2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:°*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:°2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:°2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:°*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:°2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
Ù
f
-__inference_dropout_319_layer_call_fn_8288538

inputs
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_319_layer_call_and_return_conditional_losses_82878352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
ù
f
H__inference_dropout_319_layer_call_and_return_conditional_losses_8287745

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
è
û
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8287805
input_layer_input
unknown:
°
	unknown_0:	°
	unknown_1:	°
	unknown_2:	°
	unknown_3:	°
	unknown_4:	°
	unknown_5:
°·
	unknown_6:	·
	unknown_7:	·
	unknown_8:	·
	unknown_9:	·

unknown_10:	·

unknown_11:	·

unknown_12:
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *a
f\RZ
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_82877742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameinput_layer_input
¤,
ò
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8287955

inputs'
input_layer_8287919:
°"
input_layer_8287921:	°.
batch_normalization_318_8287925:	°.
batch_normalization_318_8287927:	°.
batch_normalization_318_8287929:	°.
batch_normalization_318_8287931:	°%
dense_318_8287934:
°· 
dense_318_8287936:	·.
batch_normalization_319_8287940:	·.
batch_normalization_319_8287942:	·.
batch_normalization_319_8287944:	·.
batch_normalization_319_8287946:	·$
dense_319_8287949:	·
dense_319_8287951:
identity¢/batch_normalization_318/StatefulPartitionedCall¢/batch_normalization_319/StatefulPartitionedCall¢!dense_318/StatefulPartitionedCall¢!dense_319/StatefulPartitionedCall¢#dropout_318/StatefulPartitionedCall¢#dropout_319/StatefulPartitionedCall¢#input_layer/StatefulPartitionedCall¬
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinput_layer_8287919input_layer_8287921*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_82877012%
#input_layer/StatefulPartitionedCall¢
#dropout_318/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_318_layer_call_and_return_conditional_losses_82878682%
#dropout_318/StatefulPartitionedCallÒ
/batch_normalization_318/StatefulPartitionedCallStatefulPartitionedCall,dropout_318/StatefulPartitionedCall:output:0batch_normalization_318_8287925batch_normalization_318_8287927batch_normalization_318_8287929batch_normalization_318_8287931*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_828744321
/batch_normalization_318/StatefulPartitionedCallÔ
!dense_318/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_318/StatefulPartitionedCall:output:0dense_318_8287934dense_318_8287936*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_82877342#
!dense_318/StatefulPartitionedCallÆ
#dropout_319/StatefulPartitionedCallStatefulPartitionedCall*dense_318/StatefulPartitionedCall:output:0$^dropout_318/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_319_layer_call_and_return_conditional_losses_82878352%
#dropout_319/StatefulPartitionedCallÒ
/batch_normalization_319/StatefulPartitionedCallStatefulPartitionedCall,dropout_319/StatefulPartitionedCall:output:0batch_normalization_319_8287940batch_normalization_319_8287942batch_normalization_319_8287944batch_normalization_319_8287946*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_828760521
/batch_normalization_319/StatefulPartitionedCallÓ
!dense_319/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_319/StatefulPartitionedCall:output:0dense_319_8287949dense_319_8287951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_82877672#
!dense_319/StatefulPartitionedCall
IdentityIdentity*dense_319/StatefulPartitionedCall:output:00^batch_normalization_318/StatefulPartitionedCall0^batch_normalization_319/StatefulPartitionedCall"^dense_318/StatefulPartitionedCall"^dense_319/StatefulPartitionedCall$^dropout_318/StatefulPartitionedCall$^dropout_319/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2b
/batch_normalization_318/StatefulPartitionedCall/batch_normalization_318/StatefulPartitionedCall2b
/batch_normalization_319/StatefulPartitionedCall/batch_normalization_319/StatefulPartitionedCall2F
!dense_318/StatefulPartitionedCall!dense_318/StatefulPartitionedCall2F
!dense_319/StatefulPartitionedCall!dense_319/StatefulPartitionedCall2J
#dropout_318/StatefulPartitionedCall#dropout_318/StatefulPartitionedCall2J
#dropout_319/StatefulPartitionedCall#dropout_319/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
g
H__inference_dropout_319_layer_call_and_return_conditional_losses_8288528

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ú?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÆ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*
dtype0*
seed2ÿÿÿÿ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ8>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
È
g
H__inference_dropout_318_layer_call_and_return_conditional_losses_8287868

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Û¹?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÆ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°*
dtype0*
seed2ÿÿÿÿ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *R¸>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
³
·
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_8288558

inputs0
!batchnorm_readvariableop_resource:	·4
%batchnorm_mul_readvariableop_resource:	·2
#batchnorm_readvariableop_1_resource:	·2
#batchnorm_readvariableop_2_resource:	·
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:·*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:·2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:·2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:·*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:·2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:·*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:·2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:·*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:·2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ·: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
Ù
f
-__inference_dropout_318_layer_call_fn_8288411

inputs
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_318_layer_call_and_return_conditional_losses_82878682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
¨
¡
"__inference__wrapped_model_8287359
input_layer_inputZ
Fmodel_seq_304_0_31_439_0_18_input_layer_matmul_readvariableop_resource:
°V
Gmodel_seq_304_0_31_439_0_18_input_layer_biasadd_readvariableop_resource:	°d
Umodel_seq_304_0_31_439_0_18_batch_normalization_318_batchnorm_readvariableop_resource:	°h
Ymodel_seq_304_0_31_439_0_18_batch_normalization_318_batchnorm_mul_readvariableop_resource:	°f
Wmodel_seq_304_0_31_439_0_18_batch_normalization_318_batchnorm_readvariableop_1_resource:	°f
Wmodel_seq_304_0_31_439_0_18_batch_normalization_318_batchnorm_readvariableop_2_resource:	°X
Dmodel_seq_304_0_31_439_0_18_dense_318_matmul_readvariableop_resource:
°·T
Emodel_seq_304_0_31_439_0_18_dense_318_biasadd_readvariableop_resource:	·d
Umodel_seq_304_0_31_439_0_18_batch_normalization_319_batchnorm_readvariableop_resource:	·h
Ymodel_seq_304_0_31_439_0_18_batch_normalization_319_batchnorm_mul_readvariableop_resource:	·f
Wmodel_seq_304_0_31_439_0_18_batch_normalization_319_batchnorm_readvariableop_1_resource:	·f
Wmodel_seq_304_0_31_439_0_18_batch_normalization_319_batchnorm_readvariableop_2_resource:	·W
Dmodel_seq_304_0_31_439_0_18_dense_319_matmul_readvariableop_resource:	·S
Emodel_seq_304_0_31_439_0_18_dense_319_biasadd_readvariableop_resource:
identity¢Lmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp¢Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_1¢Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_2¢Pmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul/ReadVariableOp¢Lmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp¢Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_1¢Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_2¢Pmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul/ReadVariableOp¢<model_seq_304_0.31_439_0.18/dense_318/BiasAdd/ReadVariableOp¢;model_seq_304_0.31_439_0.18/dense_318/MatMul/ReadVariableOp¢<model_seq_304_0.31_439_0.18/dense_319/BiasAdd/ReadVariableOp¢;model_seq_304_0.31_439_0.18/dense_319/MatMul/ReadVariableOp¢>model_seq_304_0.31_439_0.18/input_layer/BiasAdd/ReadVariableOp¢=model_seq_304_0.31_439_0.18/input_layer/MatMul/ReadVariableOp
=model_seq_304_0.31_439_0.18/input_layer/MatMul/ReadVariableOpReadVariableOpFmodel_seq_304_0_31_439_0_18_input_layer_matmul_readvariableop_resource* 
_output_shapes
:
°*
dtype02?
=model_seq_304_0.31_439_0.18/input_layer/MatMul/ReadVariableOp÷
.model_seq_304_0.31_439_0.18/input_layer/MatMulMatMulinput_layer_inputEmodel_seq_304_0.31_439_0.18/input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°20
.model_seq_304_0.31_439_0.18/input_layer/MatMul
>model_seq_304_0.31_439_0.18/input_layer/BiasAdd/ReadVariableOpReadVariableOpGmodel_seq_304_0_31_439_0_18_input_layer_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02@
>model_seq_304_0.31_439_0.18/input_layer/BiasAdd/ReadVariableOp¢
/model_seq_304_0.31_439_0.18/input_layer/BiasAddBiasAdd8model_seq_304_0.31_439_0.18/input_layer/MatMul:product:0Fmodel_seq_304_0.31_439_0.18/input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°21
/model_seq_304_0.31_439_0.18/input_layer/BiasAddÑ
,model_seq_304_0.31_439_0.18/input_layer/ReluRelu8model_seq_304_0.31_439_0.18/input_layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2.
,model_seq_304_0.31_439_0.18/input_layer/Reluß
0model_seq_304_0.31_439_0.18/dropout_318/IdentityIdentity:model_seq_304_0.31_439_0.18/input_layer/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°22
0model_seq_304_0.31_439_0.18/dropout_318/Identity¯
Lmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOpReadVariableOpUmodel_seq_304_0_31_439_0_18_batch_normalization_318_batchnorm_readvariableop_resource*
_output_shapes	
:°*
dtype02N
Lmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOpÏ
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/add/yÙ
Amodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/addAddV2Tmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp:value:0Lmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/add/y:output:0*
T0*
_output_shapes	
:°2C
Amodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/add
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/RsqrtRsqrtEmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/add:z:0*
T0*
_output_shapes	
:°2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/Rsqrt»
Pmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_seq_304_0_31_439_0_18_batch_normalization_318_batchnorm_mul_readvariableop_resource*
_output_shapes	
:°*
dtype02R
Pmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul/ReadVariableOpÖ
Amodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mulMulGmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/Rsqrt:y:0Xmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:°2C
Amodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mulÆ
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul_1Mul9model_seq_304_0.31_439_0.18/dropout_318/Identity:output:0Emodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul_1µ
Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_1ReadVariableOpWmodel_seq_304_0_31_439_0_18_batch_normalization_318_batchnorm_readvariableop_1_resource*
_output_shapes	
:°*
dtype02P
Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_1Ö
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul_2MulVmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_1:value:0Emodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul:z:0*
T0*
_output_shapes	
:°2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul_2µ
Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_2ReadVariableOpWmodel_seq_304_0_31_439_0_18_batch_normalization_318_batchnorm_readvariableop_2_resource*
_output_shapes	
:°*
dtype02P
Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_2Ô
Amodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/subSubVmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_2:value:0Gmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:°2C
Amodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/subÖ
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/add_1AddV2Gmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul_1:z:0Emodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/add_1
;model_seq_304_0.31_439_0.18/dense_318/MatMul/ReadVariableOpReadVariableOpDmodel_seq_304_0_31_439_0_18_dense_318_matmul_readvariableop_resource* 
_output_shapes
:
°·*
dtype02=
;model_seq_304_0.31_439_0.18/dense_318/MatMul/ReadVariableOp§
,model_seq_304_0.31_439_0.18/dense_318/MatMulMatMulGmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/add_1:z:0Cmodel_seq_304_0.31_439_0.18/dense_318/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2.
,model_seq_304_0.31_439_0.18/dense_318/MatMulÿ
<model_seq_304_0.31_439_0.18/dense_318/BiasAdd/ReadVariableOpReadVariableOpEmodel_seq_304_0_31_439_0_18_dense_318_biasadd_readvariableop_resource*
_output_shapes	
:·*
dtype02>
<model_seq_304_0.31_439_0.18/dense_318/BiasAdd/ReadVariableOp
-model_seq_304_0.31_439_0.18/dense_318/BiasAddBiasAdd6model_seq_304_0.31_439_0.18/dense_318/MatMul:product:0Dmodel_seq_304_0.31_439_0.18/dense_318/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2/
-model_seq_304_0.31_439_0.18/dense_318/BiasAddË
*model_seq_304_0.31_439_0.18/dense_318/ReluRelu6model_seq_304_0.31_439_0.18/dense_318/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2,
*model_seq_304_0.31_439_0.18/dense_318/ReluÝ
0model_seq_304_0.31_439_0.18/dropout_319/IdentityIdentity8model_seq_304_0.31_439_0.18/dense_318/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·22
0model_seq_304_0.31_439_0.18/dropout_319/Identity¯
Lmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOpReadVariableOpUmodel_seq_304_0_31_439_0_18_batch_normalization_319_batchnorm_readvariableop_resource*
_output_shapes	
:·*
dtype02N
Lmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOpÏ
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/add/yÙ
Amodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/addAddV2Tmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp:value:0Lmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/add/y:output:0*
T0*
_output_shapes	
:·2C
Amodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/add
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/RsqrtRsqrtEmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/add:z:0*
T0*
_output_shapes	
:·2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/Rsqrt»
Pmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_seq_304_0_31_439_0_18_batch_normalization_319_batchnorm_mul_readvariableop_resource*
_output_shapes	
:·*
dtype02R
Pmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul/ReadVariableOpÖ
Amodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mulMulGmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/Rsqrt:y:0Xmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:·2C
Amodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mulÆ
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul_1Mul9model_seq_304_0.31_439_0.18/dropout_319/Identity:output:0Emodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul_1µ
Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_1ReadVariableOpWmodel_seq_304_0_31_439_0_18_batch_normalization_319_batchnorm_readvariableop_1_resource*
_output_shapes	
:·*
dtype02P
Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_1Ö
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul_2MulVmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_1:value:0Emodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul:z:0*
T0*
_output_shapes	
:·2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul_2µ
Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_2ReadVariableOpWmodel_seq_304_0_31_439_0_18_batch_normalization_319_batchnorm_readvariableop_2_resource*
_output_shapes	
:·*
dtype02P
Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_2Ô
Amodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/subSubVmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_2:value:0Gmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:·2C
Amodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/subÖ
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/add_1AddV2Gmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul_1:z:0Emodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·2E
Cmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/add_1
;model_seq_304_0.31_439_0.18/dense_319/MatMul/ReadVariableOpReadVariableOpDmodel_seq_304_0_31_439_0_18_dense_319_matmul_readvariableop_resource*
_output_shapes
:	·*
dtype02=
;model_seq_304_0.31_439_0.18/dense_319/MatMul/ReadVariableOp¦
,model_seq_304_0.31_439_0.18/dense_319/MatMulMatMulGmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/add_1:z:0Cmodel_seq_304_0.31_439_0.18/dense_319/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_seq_304_0.31_439_0.18/dense_319/MatMulþ
<model_seq_304_0.31_439_0.18/dense_319/BiasAdd/ReadVariableOpReadVariableOpEmodel_seq_304_0_31_439_0_18_dense_319_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<model_seq_304_0.31_439_0.18/dense_319/BiasAdd/ReadVariableOp
-model_seq_304_0.31_439_0.18/dense_319/BiasAddBiasAdd6model_seq_304_0.31_439_0.18/dense_319/MatMul:product:0Dmodel_seq_304_0.31_439_0.18/dense_319/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_seq_304_0.31_439_0.18/dense_319/BiasAddÓ
-model_seq_304_0.31_439_0.18/dense_319/SoftmaxSoftmax6model_seq_304_0.31_439_0.18/dense_319/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_seq_304_0.31_439_0.18/dense_319/Softmax	
IdentityIdentity7model_seq_304_0.31_439_0.18/dense_319/Softmax:softmax:0M^model_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOpO^model_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_1O^model_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_2Q^model_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul/ReadVariableOpM^model_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOpO^model_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_1O^model_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_2Q^model_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul/ReadVariableOp=^model_seq_304_0.31_439_0.18/dense_318/BiasAdd/ReadVariableOp<^model_seq_304_0.31_439_0.18/dense_318/MatMul/ReadVariableOp=^model_seq_304_0.31_439_0.18/dense_319/BiasAdd/ReadVariableOp<^model_seq_304_0.31_439_0.18/dense_319/MatMul/ReadVariableOp?^model_seq_304_0.31_439_0.18/input_layer/BiasAdd/ReadVariableOp>^model_seq_304_0.31_439_0.18/input_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2
Lmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOpLmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp2 
Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_1Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_12 
Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_2Nmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/ReadVariableOp_22¤
Pmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul/ReadVariableOpPmodel_seq_304_0.31_439_0.18/batch_normalization_318/batchnorm/mul/ReadVariableOp2
Lmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOpLmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp2 
Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_1Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_12 
Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_2Nmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/ReadVariableOp_22¤
Pmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul/ReadVariableOpPmodel_seq_304_0.31_439_0.18/batch_normalization_319/batchnorm/mul/ReadVariableOp2|
<model_seq_304_0.31_439_0.18/dense_318/BiasAdd/ReadVariableOp<model_seq_304_0.31_439_0.18/dense_318/BiasAdd/ReadVariableOp2z
;model_seq_304_0.31_439_0.18/dense_318/MatMul/ReadVariableOp;model_seq_304_0.31_439_0.18/dense_318/MatMul/ReadVariableOp2|
<model_seq_304_0.31_439_0.18/dense_319/BiasAdd/ReadVariableOp<model_seq_304_0.31_439_0.18/dense_319/BiasAdd/ReadVariableOp2z
;model_seq_304_0.31_439_0.18/dense_319/MatMul/ReadVariableOp;model_seq_304_0.31_439_0.18/dense_319/MatMul/ReadVariableOp2
>model_seq_304_0.31_439_0.18/input_layer/BiasAdd/ReadVariableOp>model_seq_304_0.31_439_0.18/input_layer/BiasAdd/ReadVariableOp2~
=model_seq_304_0.31_439_0.18/input_layer/MatMul/ReadVariableOp=model_seq_304_0.31_439_0.18/input_layer/MatMul/ReadVariableOp:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameinput_layer_input"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Á
serving_default­
P
input_layer_input;
#serving_default_input_layer_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_3190
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:õ
ù@
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"Î=
_tf_keras_sequential¯={"name": "model_seq_304_0.31_439_0.18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "model_seq_304_0.31_439_0.18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer_input"}}, {"class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_318", "trainable": true, "dtype": "float32", "rate": 0.31, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_318", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_318", "trainable": true, "dtype": "float32", "units": 439, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_319", "trainable": true, "dtype": "float32", "rate": 0.18, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_319", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_319", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128]}, "float32", "input_layer_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "model_seq_304_0.31_439_0.18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Dropout", "config": {"name": "dropout_318", "trainable": true, "dtype": "float32", "rate": 0.31, "noise_shape": null, "seed": null}, "shared_object_id": 4}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_318", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "dense_318", "trainable": true, "dtype": "float32", "units": 439, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dropout", "config": {"name": "dropout_319", "trainable": true, "dtype": "float32", "rate": 0.18, "noise_shape": null, "seed": null}, "shared_object_id": 13}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_319", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "dense_319", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 24}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.015148183330893517, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ï	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"¨
_tf_keras_layer{"name": "input_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}

	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ò
_tf_keras_layerØ{"name": "dropout_318", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_318", "trainable": true, "dtype": "float32", "rate": 0.31, "noise_shape": null, "seed": null}, "shared_object_id": 4}
Å

axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
 	keras_api
+&call_and_return_all_conditional_losses
__call__"ï
_tf_keras_layerÕ{"name": "batch_normalization_318", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_318", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 304}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 304]}}
Ù

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
+&call_and_return_all_conditional_losses
__call__"²
_tf_keras_layer{"name": "dense_318", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_318", "trainable": true, "dtype": "float32", "units": 439, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 304}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 304]}}

'	variables
(trainable_variables
)regularization_losses
*	keras_api
+&call_and_return_all_conditional_losses
__call__"ó
_tf_keras_layerÙ{"name": "dropout_319", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_319", "trainable": true, "dtype": "float32", "rate": 0.18, "noise_shape": null, "seed": null}, "shared_object_id": 13}
Ê

+axis
	,gamma
-beta
.moving_mean
/moving_variance
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+&call_and_return_all_conditional_losses
__call__"ô
_tf_keras_layerÚ{"name": "batch_normalization_319", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_319", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 439}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 439]}}
Ú

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
+&call_and_return_all_conditional_losses
__call__"³
_tf_keras_layer{"name": "dense_319", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_319", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 439}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 439]}}

:iter

;beta_1

<beta_2
	=decay
>learning_ratemrmsmtmu!mv"mw,mx-my4mz5m{v|v}v~v!v"v,v-v4v5v"
	optimizer

0
1
2
3
4
5
!6
"7
,8
-9
.10
/11
412
513"
trackable_list_wrapper
f
0
1
2
3
!4
"5
,6
-7
48
59"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
		variables
?non_trainable_variables

trainable_variables
@metrics
Alayer_regularization_losses

Blayers
regularization_losses
Clayer_metrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
&:$
°2input_layer/kernel
:°2input_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables
Dnon_trainable_variables
trainable_variables
Emetrics
Flayer_regularization_losses

Glayers
regularization_losses
Hlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables
Inon_trainable_variables
trainable_variables
Jmetrics
Klayer_regularization_losses

Llayers
regularization_losses
Mlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*°2batch_normalization_318/gamma
+:)°2batch_normalization_318/beta
4:2° (2#batch_normalization_318/moving_mean
8:6° (2'batch_normalization_318/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables
Nnon_trainable_variables
trainable_variables
Ometrics
Player_regularization_losses

Qlayers
regularization_losses
Rlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
$:"
°·2dense_318/kernel
:·2dense_318/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
#	variables
Snon_trainable_variables
$trainable_variables
Tmetrics
Ulayer_regularization_losses

Vlayers
%regularization_losses
Wlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
'	variables
Xnon_trainable_variables
(trainable_variables
Ymetrics
Zlayer_regularization_losses

[layers
)regularization_losses
\layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*·2batch_normalization_319/gamma
+:)·2batch_normalization_319/beta
4:2· (2#batch_normalization_319/moving_mean
8:6· (2'batch_normalization_319/moving_variance
<
,0
-1
.2
/3"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
0	variables
]non_trainable_variables
1trainable_variables
^metrics
_layer_regularization_losses

`layers
2regularization_losses
alayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:!	·2dense_319/kernel
:2dense_319/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
°
6	variables
bnon_trainable_variables
7trainable_variables
cmetrics
dlayer_regularization_losses

elayers
8regularization_losses
flayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
0
1
.2
/3"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ô
	itotal
	jcount
k	variables
l	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 29}

	mtotal
	ncount
o
_fn_kwargs
p	variables
q	keras_api"Í
_tf_keras_metric²{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 24}
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
m0
n1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
+:)
°2Adam/input_layer/kernel/m
$:"°2Adam/input_layer/bias/m
1:/°2$Adam/batch_normalization_318/gamma/m
0:.°2#Adam/batch_normalization_318/beta/m
):'
°·2Adam/dense_318/kernel/m
": ·2Adam/dense_318/bias/m
1:/·2$Adam/batch_normalization_319/gamma/m
0:.·2#Adam/batch_normalization_319/beta/m
(:&	·2Adam/dense_319/kernel/m
!:2Adam/dense_319/bias/m
+:)
°2Adam/input_layer/kernel/v
$:"°2Adam/input_layer/bias/v
1:/°2$Adam/batch_normalization_318/gamma/v
0:.°2#Adam/batch_normalization_318/beta/v
):'
°·2Adam/dense_318/kernel/v
": ·2Adam/dense_318/bias/v
1:/·2$Adam/batch_normalization_319/gamma/v
0:.·2#Adam/batch_normalization_319/beta/v
(:&	·2Adam/dense_319/kernel/v
!:2Adam/dense_319/bias/v
®2«
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288197
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288298
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288058
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288097À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Â2¿
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8287805
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288331
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288364
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288019À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
"__inference__wrapped_model_8287359Á
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *1¢.
,)
input_layer_inputÿÿÿÿÿÿÿÿÿ
ò2ï
H__inference_input_layer_layer_call_and_return_conditional_losses_8288375¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_input_layer_layer_call_fn_8288384¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
H__inference_dropout_318_layer_call_and_return_conditional_losses_8288389
H__inference_dropout_318_layer_call_and_return_conditional_losses_8288401´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
-__inference_dropout_318_layer_call_fn_8288406
-__inference_dropout_318_layer_call_fn_8288411´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_8288431
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_8288465´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
°2­
9__inference_batch_normalization_318_layer_call_fn_8288478
9__inference_batch_normalization_318_layer_call_fn_8288491´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_dense_318_layer_call_and_return_conditional_losses_8288502¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_318_layer_call_fn_8288511¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
H__inference_dropout_319_layer_call_and_return_conditional_losses_8288516
H__inference_dropout_319_layer_call_and_return_conditional_losses_8288528´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
-__inference_dropout_319_layer_call_fn_8288533
-__inference_dropout_319_layer_call_fn_8288538´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_8288558
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_8288592´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
°2­
9__inference_batch_normalization_319_layer_call_fn_8288605
9__inference_batch_normalization_319_layer_call_fn_8288618´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_dense_319_layer_call_and_return_conditional_losses_8288629¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_319_layer_call_fn_8288638¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÖBÓ
%__inference_signature_wrapper_8288138input_layer_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 «
"__inference__wrapped_model_8287359!"/,.-45;¢8
1¢.
,)
input_layer_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_319# 
	dense_319ÿÿÿÿÿÿÿÿÿ¼
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_8288431d4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ°
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ°
 ¼
T__inference_batch_normalization_318_layer_call_and_return_conditional_losses_8288465d4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ°
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ°
 
9__inference_batch_normalization_318_layer_call_fn_8288478W4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ°
p 
ª "ÿÿÿÿÿÿÿÿÿ°
9__inference_batch_normalization_318_layer_call_fn_8288491W4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ°
p
ª "ÿÿÿÿÿÿÿÿÿ°¼
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_8288558d/,.-4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ·
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ·
 ¼
T__inference_batch_normalization_319_layer_call_and_return_conditional_losses_8288592d./,-4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ·
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ·
 
9__inference_batch_normalization_319_layer_call_fn_8288605W/,.-4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ·
p 
ª "ÿÿÿÿÿÿÿÿÿ·
9__inference_batch_normalization_319_layer_call_fn_8288618W./,-4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ·
p
ª "ÿÿÿÿÿÿÿÿÿ·¨
F__inference_dense_318_layer_call_and_return_conditional_losses_8288502^!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ°
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ·
 
+__inference_dense_318_layer_call_fn_8288511Q!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ°
ª "ÿÿÿÿÿÿÿÿÿ·§
F__inference_dense_319_layer_call_and_return_conditional_losses_8288629]450¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ·
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_319_layer_call_fn_8288638P450¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ·
ª "ÿÿÿÿÿÿÿÿÿª
H__inference_dropout_318_layer_call_and_return_conditional_losses_8288389^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ°
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ°
 ª
H__inference_dropout_318_layer_call_and_return_conditional_losses_8288401^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ°
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ°
 
-__inference_dropout_318_layer_call_fn_8288406Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ°
p 
ª "ÿÿÿÿÿÿÿÿÿ°
-__inference_dropout_318_layer_call_fn_8288411Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ°
p
ª "ÿÿÿÿÿÿÿÿÿ°ª
H__inference_dropout_319_layer_call_and_return_conditional_losses_8288516^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ·
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ·
 ª
H__inference_dropout_319_layer_call_and_return_conditional_losses_8288528^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ·
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ·
 
-__inference_dropout_319_layer_call_fn_8288533Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ·
p 
ª "ÿÿÿÿÿÿÿÿÿ·
-__inference_dropout_319_layer_call_fn_8288538Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ·
p
ª "ÿÿÿÿÿÿÿÿÿ·ª
H__inference_input_layer_layer_call_and_return_conditional_losses_8288375^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ°
 
-__inference_input_layer_layer_call_fn_8288384Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°Ø
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288058|!"/,.-45C¢@
9¢6
,)
input_layer_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ø
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288097|!"./,-45C¢@
9¢6
,)
input_layer_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Í
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288197q!"/,.-458¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Í
X__inference_model_seq_304_0.31_439_0.18_layer_call_and_return_conditional_losses_8288298q!"./,-458¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8287805o!"/,.-45C¢@
9¢6
,)
input_layer_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ°
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288019o!"./,-45C¢@
9¢6
,)
input_layer_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¥
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288331d!"/,.-458¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¥
=__inference_model_seq_304_0.31_439_0.18_layer_call_fn_8288364d!"./,-458¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÃ
%__inference_signature_wrapper_8288138!"/,.-45P¢M
¢ 
FªC
A
input_layer_input,)
input_layer_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_319# 
	dense_319ÿÿÿÿÿÿÿÿÿ
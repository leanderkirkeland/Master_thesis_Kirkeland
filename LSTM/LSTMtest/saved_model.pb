ÆÏ2
®!ÿ 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68é¼0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:
*
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

conv_lstm2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv_lstm2d/kernel

&conv_lstm2d/kernel/Read/ReadVariableOpReadVariableOpconv_lstm2d/kernel*&
_output_shapes
: *
dtype0

conv_lstm2d/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameconv_lstm2d/recurrent_kernel

0conv_lstm2d/recurrent_kernel/Read/ReadVariableOpReadVariableOpconv_lstm2d/recurrent_kernel*&
_output_shapes
: *
dtype0
x
conv_lstm2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv_lstm2d/bias
q
$conv_lstm2d/bias/Read/ReadVariableOpReadVariableOpconv_lstm2d/bias*
_output_shapes
: *
dtype0

conv_lstm2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv_lstm2d_1/kernel

(conv_lstm2d_1/kernel/Read/ReadVariableOpReadVariableOpconv_lstm2d_1/kernel*&
_output_shapes
:@*
dtype0
 
conv_lstm2d_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name conv_lstm2d_1/recurrent_kernel

2conv_lstm2d_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpconv_lstm2d_1/recurrent_kernel*&
_output_shapes
:@*
dtype0
|
conv_lstm2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameconv_lstm2d_1/bias
u
&conv_lstm2d_1/bias/Read/ReadVariableOpReadVariableOpconv_lstm2d_1/bias*
_output_shapes
:@*
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

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:
*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:
*
dtype0

Adam/conv_lstm2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv_lstm2d/kernel/m

-Adam/conv_lstm2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d/kernel/m*&
_output_shapes
: *
dtype0
ª
#Adam/conv_lstm2d/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/conv_lstm2d/recurrent_kernel/m
£
7Adam/conv_lstm2d/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp#Adam/conv_lstm2d/recurrent_kernel/m*&
_output_shapes
: *
dtype0

Adam/conv_lstm2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv_lstm2d/bias/m

+Adam/conv_lstm2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d/bias/m*
_output_shapes
: *
dtype0

Adam/conv_lstm2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv_lstm2d_1/kernel/m

/Adam/conv_lstm2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d_1/kernel/m*&
_output_shapes
:@*
dtype0
®
%Adam/conv_lstm2d_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/conv_lstm2d_1/recurrent_kernel/m
§
9Adam/conv_lstm2d_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_lstm2d_1/recurrent_kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv_lstm2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/conv_lstm2d_1/bias/m

-Adam/conv_lstm2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d_1/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:
*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:
*
dtype0

Adam/conv_lstm2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv_lstm2d/kernel/v

-Adam/conv_lstm2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d/kernel/v*&
_output_shapes
: *
dtype0
ª
#Adam/conv_lstm2d/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/conv_lstm2d/recurrent_kernel/v
£
7Adam/conv_lstm2d/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp#Adam/conv_lstm2d/recurrent_kernel/v*&
_output_shapes
: *
dtype0

Adam/conv_lstm2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv_lstm2d/bias/v

+Adam/conv_lstm2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d/bias/v*
_output_shapes
: *
dtype0

Adam/conv_lstm2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv_lstm2d_1/kernel/v

/Adam/conv_lstm2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d_1/kernel/v*&
_output_shapes
:@*
dtype0
®
%Adam/conv_lstm2d_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/conv_lstm2d_1/recurrent_kernel/v
§
9Adam/conv_lstm2d_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_lstm2d_1/recurrent_kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv_lstm2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/conv_lstm2d_1/bias/v

-Adam/conv_lstm2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d_1/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*½U
value³UB°U B©U

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
ª
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
ª
 cell
!
state_spec
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*

	(layer
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 
¦

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
¦

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*

?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate/m©0mª7m«8m¬Dm­Em®Fm¯Gm°Hm±Im²/v³0v´7vµ8v¶Dv·Ev¸Fv¹GvºHv»Iv¼*
J
D0
E1
F2
G3
H4
I5
/6
07
78
89*
J
D0
E1
F2
G3
H4
I5
/6
07
78
89*
* 
°
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Oserving_default* 
Ó

Dkernel
Erecurrent_kernel
Fbias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T_random_generator
U__call__
*V&call_and_return_all_conditional_losses*
* 

D0
E1
F2*

D0
E1
F2*
* 


Wstates
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 

]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
* 
* 
* 

cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
Ó

Gkernel
Hrecurrent_kernel
Ibias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l_random_generator
m__call__
*n&call_and_return_all_conditional_losses*
* 

G0
H1
I2*

G0
H1
I2*
* 


ostates
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 

u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses* 
* 
* 
* 

{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEconv_lstm2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv_lstm2d/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv_lstm2d/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv_lstm2d_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_lstm2d_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEconv_lstm2d_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

0
1*
* 
* 
* 

D0
E1
F2*

D0
E1
F2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 
* 
* 
* 
	
0* 
* 
* 
* 

G0
H1
I2*

G0
H1
I2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
h	variables
itrainable_variables
jregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

 0*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 
* 
* 
* 
	
(0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

 total

¡count
¢	variables
£	keras_api*
M

¤total

¥count
¦
_fn_kwargs
§	variables
¨	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

 0
¡1*

¢	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¤0
¥1*

§	variables*
z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv_lstm2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_lstm2d/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv_lstm2d/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/conv_lstm2d_1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/conv_lstm2d_1/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv_lstm2d_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv_lstm2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/conv_lstm2d/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv_lstm2d/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/conv_lstm2d_1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/conv_lstm2d_1/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv_lstm2d_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *
dtype0*(
shape:ÿÿÿÿÿÿÿÿÿd@ 

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_lstm2d/kernelconv_lstm2d/recurrent_kernelconv_lstm2d/biasconv_lstm2d_1/kernelconv_lstm2d_1/recurrent_kernelconv_lstm2d_1/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_172333
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp&conv_lstm2d/kernel/Read/ReadVariableOp0conv_lstm2d/recurrent_kernel/Read/ReadVariableOp$conv_lstm2d/bias/Read/ReadVariableOp(conv_lstm2d_1/kernel/Read/ReadVariableOp2conv_lstm2d_1/recurrent_kernel/Read/ReadVariableOp&conv_lstm2d_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp-Adam/conv_lstm2d/kernel/m/Read/ReadVariableOp7Adam/conv_lstm2d/recurrent_kernel/m/Read/ReadVariableOp+Adam/conv_lstm2d/bias/m/Read/ReadVariableOp/Adam/conv_lstm2d_1/kernel/m/Read/ReadVariableOp9Adam/conv_lstm2d_1/recurrent_kernel/m/Read/ReadVariableOp-Adam/conv_lstm2d_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp-Adam/conv_lstm2d/kernel/v/Read/ReadVariableOp7Adam/conv_lstm2d/recurrent_kernel/v/Read/ReadVariableOp+Adam/conv_lstm2d/bias/v/Read/ReadVariableOp/Adam/conv_lstm2d_1/kernel/v/Read/ReadVariableOp9Adam/conv_lstm2d_1/recurrent_kernel/v/Read/ReadVariableOp-Adam/conv_lstm2d_1/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_174906
	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv_lstm2d/kernelconv_lstm2d/recurrent_kernelconv_lstm2d/biasconv_lstm2d_1/kernelconv_lstm2d_1/recurrent_kernelconv_lstm2d_1/biastotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv_lstm2d/kernel/m#Adam/conv_lstm2d/recurrent_kernel/mAdam/conv_lstm2d/bias/mAdam/conv_lstm2d_1/kernel/m%Adam/conv_lstm2d_1/recurrent_kernel/mAdam/conv_lstm2d_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv_lstm2d/kernel/v#Adam/conv_lstm2d/recurrent_kernel/vAdam/conv_lstm2d/bias/vAdam/conv_lstm2d_1/kernel/v%Adam/conv_lstm2d_1/recurrent_kernel/vAdam/conv_lstm2d_1/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_175033ø.
Ñ
Á
while_cond_172910
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_172910___redundant_placeholder04
0while_while_cond_172910___redundant_placeholder14
0while_while_cond_172910___redundant_placeholder24
0while_while_cond_172910___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
ù

'__inference_conv2d_layer_call_fn_174296

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_170472{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
þc
Ü
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_170796

inputs7
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢while]

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    t
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                u
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_170670*
condR
while_cond_170669*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          Ê
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd j
IdentityIdentitytranspose_1:y:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
 
_user_specified_nameinputs
¶
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_173285

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B : S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_169919

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿!

while_body_169599
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_169623_0:@(
while_169625_0:@
while_169627_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_169623:@&
while_169625:@
while_169627:@¢while/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_169623_0while_169625_0while_169627_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_169585Ï
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
while_169623while_169623_0"
while_169625while_169625_0"
while_169627while_169627_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
t
«
conv_lstm2d_1_while_body_1721128
4conv_lstm2d_1_while_conv_lstm2d_1_while_loop_counter>
:conv_lstm2d_1_while_conv_lstm2d_1_while_maximum_iterations#
conv_lstm2d_1_while_placeholder%
!conv_lstm2d_1_while_placeholder_1%
!conv_lstm2d_1_while_placeholder_2%
!conv_lstm2d_1_while_placeholder_35
1conv_lstm2d_1_while_conv_lstm2d_1_strided_slice_0s
oconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0M
3conv_lstm2d_1_while_split_readvariableop_resource_0:@O
5conv_lstm2d_1_while_split_1_readvariableop_resource_0:@C
5conv_lstm2d_1_while_split_2_readvariableop_resource_0:@ 
conv_lstm2d_1_while_identity"
conv_lstm2d_1_while_identity_1"
conv_lstm2d_1_while_identity_2"
conv_lstm2d_1_while_identity_3"
conv_lstm2d_1_while_identity_4"
conv_lstm2d_1_while_identity_53
/conv_lstm2d_1_while_conv_lstm2d_1_strided_sliceq
mconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensorK
1conv_lstm2d_1_while_split_readvariableop_resource:@M
3conv_lstm2d_1_while_split_1_readvariableop_resource:@A
3conv_lstm2d_1_while_split_2_readvariableop_resource:@¢(conv_lstm2d_1/while/split/ReadVariableOp¢*conv_lstm2d_1/while/split_1/ReadVariableOp¢*conv_lstm2d_1/while/split_2/ReadVariableOp
Econv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ô
7conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0conv_lstm2d_1_while_placeholderNconv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0e
#conv_lstm2d_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¤
(conv_lstm2d_1/while/split/ReadVariableOpReadVariableOp3conv_lstm2d_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0ú
conv_lstm2d_1/while/splitSplit,conv_lstm2d_1/while/split/split_dim:output:00conv_lstm2d_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitg
%conv_lstm2d_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
*conv_lstm2d_1/while/split_1/ReadVariableOpReadVariableOp5conv_lstm2d_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0
conv_lstm2d_1/while/split_1Split.conv_lstm2d_1/while/split_1/split_dim:output:02conv_lstm2d_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitg
%conv_lstm2d_1/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
*conv_lstm2d_1/while/split_2/ReadVariableOpReadVariableOp5conv_lstm2d_1_while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ð
conv_lstm2d_1/while/split_2Split.conv_lstm2d_1/while/split_2/split_dim:output:02conv_lstm2d_1/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitï
conv_lstm2d_1/while/convolutionConv2D>conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_1/while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
°
conv_lstm2d_1/while/BiasAddBiasAdd(conv_lstm2d_1/while/convolution:output:0$conv_lstm2d_1/while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ñ
!conv_lstm2d_1/while/convolution_1Conv2D>conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_1/while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
conv_lstm2d_1/while/BiasAdd_1BiasAdd*conv_lstm2d_1/while/convolution_1:output:0$conv_lstm2d_1/while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ñ
!conv_lstm2d_1/while/convolution_2Conv2D>conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_1/while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
conv_lstm2d_1/while/BiasAdd_2BiasAdd*conv_lstm2d_1/while/convolution_2:output:0$conv_lstm2d_1/while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ñ
!conv_lstm2d_1/while/convolution_3Conv2D>conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_1/while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
conv_lstm2d_1/while/BiasAdd_3BiasAdd*conv_lstm2d_1/while/convolution_3:output:0$conv_lstm2d_1/while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
!conv_lstm2d_1/while/convolution_4Conv2D!conv_lstm2d_1_while_placeholder_2$conv_lstm2d_1/while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ö
!conv_lstm2d_1/while/convolution_5Conv2D!conv_lstm2d_1_while_placeholder_2$conv_lstm2d_1/while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ö
!conv_lstm2d_1/while/convolution_6Conv2D!conv_lstm2d_1_while_placeholder_2$conv_lstm2d_1/while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ö
!conv_lstm2d_1/while/convolution_7Conv2D!conv_lstm2d_1_while_placeholder_2$conv_lstm2d_1/while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
conv_lstm2d_1/while/addAddV2$conv_lstm2d_1/while/BiasAdd:output:0*conv_lstm2d_1/while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
conv_lstm2d_1/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
conv_lstm2d_1/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/while/MulMulconv_lstm2d_1/while/add:z:0"conv_lstm2d_1/while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/while/Add_1AddV2conv_lstm2d_1/while/Mul:z:0$conv_lstm2d_1/while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
+conv_lstm2d_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
)conv_lstm2d_1/while/clip_by_value/MinimumMinimumconv_lstm2d_1/while/Add_1:z:04conv_lstm2d_1/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
#conv_lstm2d_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!conv_lstm2d_1/while/clip_by_valueMaximum-conv_lstm2d_1/while/clip_by_value/Minimum:z:0,conv_lstm2d_1/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
conv_lstm2d_1/while/add_2AddV2&conv_lstm2d_1/while/BiasAdd_1:output:0*conv_lstm2d_1/while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
conv_lstm2d_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
conv_lstm2d_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/while/Mul_1Mulconv_lstm2d_1/while/add_2:z:0$conv_lstm2d_1/while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
conv_lstm2d_1/while/Add_3AddV2conv_lstm2d_1/while/Mul_1:z:0$conv_lstm2d_1/while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
-conv_lstm2d_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
+conv_lstm2d_1/while/clip_by_value_1/MinimumMinimumconv_lstm2d_1/while/Add_3:z:06conv_lstm2d_1/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
%conv_lstm2d_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    É
#conv_lstm2d_1/while/clip_by_value_1Maximum/conv_lstm2d_1/while/clip_by_value_1/Minimum:z:0.conv_lstm2d_1/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
conv_lstm2d_1/while/mul_2Mul'conv_lstm2d_1/while/clip_by_value_1:z:0!conv_lstm2d_1_while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
conv_lstm2d_1/while/add_4AddV2&conv_lstm2d_1/while/BiasAdd_2:output:0*conv_lstm2d_1/while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
conv_lstm2d_1/while/ReluReluconv_lstm2d_1/while/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
conv_lstm2d_1/while/mul_3Mul%conv_lstm2d_1/while/clip_by_value:z:0&conv_lstm2d_1/while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/while/add_5AddV2conv_lstm2d_1/while/mul_2:z:0conv_lstm2d_1/while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
conv_lstm2d_1/while/add_6AddV2&conv_lstm2d_1/while/BiasAdd_3:output:0*conv_lstm2d_1/while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
conv_lstm2d_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
conv_lstm2d_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/while/Mul_4Mulconv_lstm2d_1/while/add_6:z:0$conv_lstm2d_1/while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
conv_lstm2d_1/while/Add_7AddV2conv_lstm2d_1/while/Mul_4:z:0$conv_lstm2d_1/while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
-conv_lstm2d_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
+conv_lstm2d_1/while/clip_by_value_2/MinimumMinimumconv_lstm2d_1/while/Add_7:z:06conv_lstm2d_1/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
%conv_lstm2d_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    É
#conv_lstm2d_1/while/clip_by_value_2Maximum/conv_lstm2d_1/while/clip_by_value_2/Minimum:z:0.conv_lstm2d_1/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
conv_lstm2d_1/while/Relu_1Reluconv_lstm2d_1/while/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
conv_lstm2d_1/while/mul_5Mul'conv_lstm2d_1/while/clip_by_value_2:z:0(conv_lstm2d_1/while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
8conv_lstm2d_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!conv_lstm2d_1_while_placeholder_1conv_lstm2d_1_while_placeholderconv_lstm2d_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒ]
conv_lstm2d_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d_1/while/add_8AddV2conv_lstm2d_1_while_placeholder$conv_lstm2d_1/while/add_8/y:output:0*
T0*
_output_shapes
: ]
conv_lstm2d_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d_1/while/add_9AddV24conv_lstm2d_1_while_conv_lstm2d_1_while_loop_counter$conv_lstm2d_1/while/add_9/y:output:0*
T0*
_output_shapes
: 
conv_lstm2d_1/while/IdentityIdentityconv_lstm2d_1/while/add_9:z:0^conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: ¢
conv_lstm2d_1/while/Identity_1Identity:conv_lstm2d_1_while_conv_lstm2d_1_while_maximum_iterations^conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: 
conv_lstm2d_1/while/Identity_2Identityconv_lstm2d_1/while/add_8:z:0^conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: Ã
conv_lstm2d_1/while/Identity_3IdentityHconv_lstm2d_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: :éèÒ
conv_lstm2d_1/while/Identity_4Identityconv_lstm2d_1/while/mul_5:z:0^conv_lstm2d_1/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/while/Identity_5Identityconv_lstm2d_1/while/add_5:z:0^conv_lstm2d_1/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß
conv_lstm2d_1/while/NoOpNoOp)^conv_lstm2d_1/while/split/ReadVariableOp+^conv_lstm2d_1/while/split_1/ReadVariableOp+^conv_lstm2d_1/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/conv_lstm2d_1_while_conv_lstm2d_1_strided_slice1conv_lstm2d_1_while_conv_lstm2d_1_strided_slice_0"E
conv_lstm2d_1_while_identity%conv_lstm2d_1/while/Identity:output:0"I
conv_lstm2d_1_while_identity_1'conv_lstm2d_1/while/Identity_1:output:0"I
conv_lstm2d_1_while_identity_2'conv_lstm2d_1/while/Identity_2:output:0"I
conv_lstm2d_1_while_identity_3'conv_lstm2d_1/while/Identity_3:output:0"I
conv_lstm2d_1_while_identity_4'conv_lstm2d_1/while/Identity_4:output:0"I
conv_lstm2d_1_while_identity_5'conv_lstm2d_1/while/Identity_5:output:0"l
3conv_lstm2d_1_while_split_1_readvariableop_resource5conv_lstm2d_1_while_split_1_readvariableop_resource_0"l
3conv_lstm2d_1_while_split_2_readvariableop_resource5conv_lstm2d_1_while_split_2_readvariableop_resource_0"h
1conv_lstm2d_1_while_split_readvariableop_resource3conv_lstm2d_1_while_split_readvariableop_resource_0"à
mconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensoroconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2T
(conv_lstm2d_1/while/split/ReadVariableOp(conv_lstm2d_1/while/split/ReadVariableOp2X
*conv_lstm2d_1/while/split_1/ReadVariableOp*conv_lstm2d_1/while/split_1/ReadVariableOp2X
*conv_lstm2d_1/while/split_2/ReadVariableOp*conv_lstm2d_1/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ñ
Á
while_cond_170069
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_170069___redundant_placeholder04
0while_while_cond_170069___redundant_placeholder14
0while_while_cond_170069___redundant_placeholder24
0while_while_cond_170069___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
ö
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_169455

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ×
max_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_169430\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B : S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&max_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs

ë
%model_conv_lstm2d_1_while_body_168809D
@model_conv_lstm2d_1_while_model_conv_lstm2d_1_while_loop_counterJ
Fmodel_conv_lstm2d_1_while_model_conv_lstm2d_1_while_maximum_iterations)
%model_conv_lstm2d_1_while_placeholder+
'model_conv_lstm2d_1_while_placeholder_1+
'model_conv_lstm2d_1_while_placeholder_2+
'model_conv_lstm2d_1_while_placeholder_3A
=model_conv_lstm2d_1_while_model_conv_lstm2d_1_strided_slice_0
{model_conv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0S
9model_conv_lstm2d_1_while_split_readvariableop_resource_0:@U
;model_conv_lstm2d_1_while_split_1_readvariableop_resource_0:@I
;model_conv_lstm2d_1_while_split_2_readvariableop_resource_0:@&
"model_conv_lstm2d_1_while_identity(
$model_conv_lstm2d_1_while_identity_1(
$model_conv_lstm2d_1_while_identity_2(
$model_conv_lstm2d_1_while_identity_3(
$model_conv_lstm2d_1_while_identity_4(
$model_conv_lstm2d_1_while_identity_5?
;model_conv_lstm2d_1_while_model_conv_lstm2d_1_strided_slice}
ymodel_conv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensorQ
7model_conv_lstm2d_1_while_split_readvariableop_resource:@S
9model_conv_lstm2d_1_while_split_1_readvariableop_resource:@G
9model_conv_lstm2d_1_while_split_2_readvariableop_resource:@¢.model/conv_lstm2d_1/while/split/ReadVariableOp¢0model/conv_lstm2d_1/while/split_1/ReadVariableOp¢0model/conv_lstm2d_1/while/split_2/ReadVariableOp¤
Kmodel/conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          
=model/conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{model_conv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0%model_conv_lstm2d_1_while_placeholderTmodel/conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0k
)model/conv_lstm2d_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :°
.model/conv_lstm2d_1/while/split/ReadVariableOpReadVariableOp9model_conv_lstm2d_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0
model/conv_lstm2d_1/while/splitSplit2model/conv_lstm2d_1/while/split/split_dim:output:06model/conv_lstm2d_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitm
+model/conv_lstm2d_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :´
0model/conv_lstm2d_1/while/split_1/ReadVariableOpReadVariableOp;model_conv_lstm2d_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0
!model/conv_lstm2d_1/while/split_1Split4model/conv_lstm2d_1/while/split_1/split_dim:output:08model/conv_lstm2d_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitm
+model/conv_lstm2d_1/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ¨
0model/conv_lstm2d_1/while/split_2/ReadVariableOpReadVariableOp;model_conv_lstm2d_1_while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0â
!model/conv_lstm2d_1/while/split_2Split4model/conv_lstm2d_1/while/split_2/split_dim:output:08model/conv_lstm2d_1/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
%model/conv_lstm2d_1/while/convolutionConv2DDmodel/conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0(model/conv_lstm2d_1/while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Â
!model/conv_lstm2d_1/while/BiasAddBiasAdd.model/conv_lstm2d_1/while/convolution:output:0*model/conv_lstm2d_1/while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'model/conv_lstm2d_1/while/convolution_1Conv2DDmodel/conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0(model/conv_lstm2d_1/while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Æ
#model/conv_lstm2d_1/while/BiasAdd_1BiasAdd0model/conv_lstm2d_1/while/convolution_1:output:0*model/conv_lstm2d_1/while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'model/conv_lstm2d_1/while/convolution_2Conv2DDmodel/conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0(model/conv_lstm2d_1/while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Æ
#model/conv_lstm2d_1/while/BiasAdd_2BiasAdd0model/conv_lstm2d_1/while/convolution_2:output:0*model/conv_lstm2d_1/while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'model/conv_lstm2d_1/while/convolution_3Conv2DDmodel/conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0(model/conv_lstm2d_1/while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Æ
#model/conv_lstm2d_1/while/BiasAdd_3BiasAdd0model/conv_lstm2d_1/while/convolution_3:output:0*model/conv_lstm2d_1/while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ è
'model/conv_lstm2d_1/while/convolution_4Conv2D'model_conv_lstm2d_1_while_placeholder_2*model/conv_lstm2d_1/while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
è
'model/conv_lstm2d_1/while/convolution_5Conv2D'model_conv_lstm2d_1_while_placeholder_2*model/conv_lstm2d_1/while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
è
'model/conv_lstm2d_1/while/convolution_6Conv2D'model_conv_lstm2d_1_while_placeholder_2*model/conv_lstm2d_1/while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
è
'model/conv_lstm2d_1/while/convolution_7Conv2D'model_conv_lstm2d_1_while_placeholder_2*model/conv_lstm2d_1/while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¾
model/conv_lstm2d_1/while/addAddV2*model/conv_lstm2d_1/while/BiasAdd:output:00model/conv_lstm2d_1/while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
model/conv_lstm2d_1/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>f
!model/conv_lstm2d_1/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?«
model/conv_lstm2d_1/while/MulMul!model/conv_lstm2d_1/while/add:z:0(model/conv_lstm2d_1/while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±
model/conv_lstm2d_1/while/Add_1AddV2!model/conv_lstm2d_1/while/Mul:z:0*model/conv_lstm2d_1/while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
1model/conv_lstm2d_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Õ
/model/conv_lstm2d_1/while/clip_by_value/MinimumMinimum#model/conv_lstm2d_1/while/Add_1:z:0:model/conv_lstm2d_1/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
)model/conv_lstm2d_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Õ
'model/conv_lstm2d_1/while/clip_by_valueMaximum3model/conv_lstm2d_1/while/clip_by_value/Minimum:z:02model/conv_lstm2d_1/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Â
model/conv_lstm2d_1/while/add_2AddV2,model/conv_lstm2d_1/while/BiasAdd_1:output:00model/conv_lstm2d_1/while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
!model/conv_lstm2d_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>f
!model/conv_lstm2d_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?±
model/conv_lstm2d_1/while/Mul_1Mul#model/conv_lstm2d_1/while/add_2:z:0*model/conv_lstm2d_1/while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
model/conv_lstm2d_1/while/Add_3AddV2#model/conv_lstm2d_1/while/Mul_1:z:0*model/conv_lstm2d_1/while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
3model/conv_lstm2d_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ù
1model/conv_lstm2d_1/while/clip_by_value_1/MinimumMinimum#model/conv_lstm2d_1/while/Add_3:z:0<model/conv_lstm2d_1/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
+model/conv_lstm2d_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Û
)model/conv_lstm2d_1/while/clip_by_value_1Maximum5model/conv_lstm2d_1/while/clip_by_value_1/Minimum:z:04model/conv_lstm2d_1/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
model/conv_lstm2d_1/while/mul_2Mul-model/conv_lstm2d_1/while/clip_by_value_1:z:0'model_conv_lstm2d_1_while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Â
model/conv_lstm2d_1/while/add_4AddV2,model/conv_lstm2d_1/while/BiasAdd_2:output:00model/conv_lstm2d_1/while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model/conv_lstm2d_1/while/ReluRelu#model/conv_lstm2d_1/while/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ »
model/conv_lstm2d_1/while/mul_3Mul+model/conv_lstm2d_1/while/clip_by_value:z:0,model/conv_lstm2d_1/while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
model/conv_lstm2d_1/while/add_5AddV2#model/conv_lstm2d_1/while/mul_2:z:0#model/conv_lstm2d_1/while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Â
model/conv_lstm2d_1/while/add_6AddV2,model/conv_lstm2d_1/while/BiasAdd_3:output:00model/conv_lstm2d_1/while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
!model/conv_lstm2d_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>f
!model/conv_lstm2d_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?±
model/conv_lstm2d_1/while/Mul_4Mul#model/conv_lstm2d_1/while/add_6:z:0*model/conv_lstm2d_1/while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
model/conv_lstm2d_1/while/Add_7AddV2#model/conv_lstm2d_1/while/Mul_4:z:0*model/conv_lstm2d_1/while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
3model/conv_lstm2d_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ù
1model/conv_lstm2d_1/while/clip_by_value_2/MinimumMinimum#model/conv_lstm2d_1/while/Add_7:z:0<model/conv_lstm2d_1/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
+model/conv_lstm2d_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Û
)model/conv_lstm2d_1/while/clip_by_value_2Maximum5model/conv_lstm2d_1/while/clip_by_value_2/Minimum:z:04model/conv_lstm2d_1/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 model/conv_lstm2d_1/while/Relu_1Relu#model/conv_lstm2d_1/while/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¿
model/conv_lstm2d_1/while/mul_5Mul-model/conv_lstm2d_1/while/clip_by_value_2:z:0.model/conv_lstm2d_1/while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
>model/conv_lstm2d_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'model_conv_lstm2d_1_while_placeholder_1%model_conv_lstm2d_1_while_placeholder#model/conv_lstm2d_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒc
!model/conv_lstm2d_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :
model/conv_lstm2d_1/while/add_8AddV2%model_conv_lstm2d_1_while_placeholder*model/conv_lstm2d_1/while/add_8/y:output:0*
T0*
_output_shapes
: c
!model/conv_lstm2d_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :·
model/conv_lstm2d_1/while/add_9AddV2@model_conv_lstm2d_1_while_model_conv_lstm2d_1_while_loop_counter*model/conv_lstm2d_1/while/add_9/y:output:0*
T0*
_output_shapes
: 
"model/conv_lstm2d_1/while/IdentityIdentity#model/conv_lstm2d_1/while/add_9:z:0^model/conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: º
$model/conv_lstm2d_1/while/Identity_1IdentityFmodel_conv_lstm2d_1_while_model_conv_lstm2d_1_while_maximum_iterations^model/conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: 
$model/conv_lstm2d_1/while/Identity_2Identity#model/conv_lstm2d_1/while/add_8:z:0^model/conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: Õ
$model/conv_lstm2d_1/while/Identity_3IdentityNmodel/conv_lstm2d_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: :éèÒ°
$model/conv_lstm2d_1/while/Identity_4Identity#model/conv_lstm2d_1/while/mul_5:z:0^model/conv_lstm2d_1/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
$model/conv_lstm2d_1/while/Identity_5Identity#model/conv_lstm2d_1/while/add_5:z:0^model/conv_lstm2d_1/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ÷
model/conv_lstm2d_1/while/NoOpNoOp/^model/conv_lstm2d_1/while/split/ReadVariableOp1^model/conv_lstm2d_1/while/split_1/ReadVariableOp1^model/conv_lstm2d_1/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"model_conv_lstm2d_1_while_identity+model/conv_lstm2d_1/while/Identity:output:0"U
$model_conv_lstm2d_1_while_identity_1-model/conv_lstm2d_1/while/Identity_1:output:0"U
$model_conv_lstm2d_1_while_identity_2-model/conv_lstm2d_1/while/Identity_2:output:0"U
$model_conv_lstm2d_1_while_identity_3-model/conv_lstm2d_1/while/Identity_3:output:0"U
$model_conv_lstm2d_1_while_identity_4-model/conv_lstm2d_1/while/Identity_4:output:0"U
$model_conv_lstm2d_1_while_identity_5-model/conv_lstm2d_1/while/Identity_5:output:0"|
;model_conv_lstm2d_1_while_model_conv_lstm2d_1_strided_slice=model_conv_lstm2d_1_while_model_conv_lstm2d_1_strided_slice_0"x
9model_conv_lstm2d_1_while_split_1_readvariableop_resource;model_conv_lstm2d_1_while_split_1_readvariableop_resource_0"x
9model_conv_lstm2d_1_while_split_2_readvariableop_resource;model_conv_lstm2d_1_while_split_2_readvariableop_resource_0"t
7model_conv_lstm2d_1_while_split_readvariableop_resource9model_conv_lstm2d_1_while_split_readvariableop_resource_0"ø
ymodel_conv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor{model_conv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2`
.model/conv_lstm2d_1/while/split/ReadVariableOp.model/conv_lstm2d_1/while/split/ReadVariableOp2d
0model/conv_lstm2d_1/while/split_1/ReadVariableOp0model/conv_lstm2d_1/while/split_1/ReadVariableOp2d
0model/conv_lstm2d_1/while/split_2/ReadVariableOp0model/conv_lstm2d_1/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
3
ë
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_169667

inputs!
unknown:@#
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCall¢whilef

zeros_like	ZerosLikeinputs*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    t
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ~
	transpose	Transposeinputstranspose/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskñ
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_169585v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_169599*
condR
while_cond_169598*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          Ó
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                §
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
IdentityIdentitytranspose_1:y:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ h
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ
Á
while_cond_170911
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_170911___redundant_placeholder04
0while_while_cond_170911___redundant_placeholder14
0while_while_cond_170911___redundant_placeholder24
0while_while_cond_170911___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
Ñ
Á
while_cond_173886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_173886___redundant_placeholder04
0while_while_cond_173886___redundant_placeholder14
0while_while_cond_173886___redundant_placeholder24
0while_while_cond_173886___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
â
È
,__inference_conv_lstm2d_layer_call_fn_172355
inputs_0!
unknown: #
	unknown_0: 
	unknown_1: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_169412
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
inputs/0
¯Ò
Ú	
A__inference_model_layer_call_and_return_conditional_losses_172306

inputsC
)conv_lstm2d_split_readvariableop_resource: E
+conv_lstm2d_split_1_readvariableop_resource: 9
+conv_lstm2d_split_2_readvariableop_resource: E
+conv_lstm2d_1_split_readvariableop_resource:@G
-conv_lstm2d_1_split_1_readvariableop_resource:@;
-conv_lstm2d_1_split_2_readvariableop_resource:@F
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:
I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:

identity¢#conv2d/Conv2D/Conv2D/ReadVariableOp¢0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_1/Conv2D/Conv2D/ReadVariableOp¢2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp¢ conv_lstm2d/split/ReadVariableOp¢"conv_lstm2d/split_1/ReadVariableOp¢"conv_lstm2d/split_2/ReadVariableOp¢conv_lstm2d/while¢"conv_lstm2d_1/split/ReadVariableOp¢$conv_lstm2d_1/split_1/ReadVariableOp¢$conv_lstm2d_1/split_2/ReadVariableOp¢conv_lstm2d_1/whilei
conv_lstm2d/zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ c
!conv_lstm2d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d/SumSumconv_lstm2d/zeros_like:y:0*conv_lstm2d/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
conv_lstm2d/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    ¹
conv_lstm2d/convolutionConv2Dconv_lstm2d/Sum:output:0conv_lstm2d/zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
w
conv_lstm2d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
conv_lstm2d/transpose	Transposeinputs#conv_lstm2d/transpose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ Z
conv_lstm2d/ShapeShapeconv_lstm2d/transpose:y:0*
T0*
_output_shapes
:i
conv_lstm2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!conv_lstm2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!conv_lstm2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv_lstm2d/strided_sliceStridedSliceconv_lstm2d/Shape:output:0(conv_lstm2d/strided_slice/stack:output:0*conv_lstm2d/strided_slice/stack_1:output:0*conv_lstm2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'conv_lstm2d/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
conv_lstm2d/TensorArrayV2TensorListReserve0conv_lstm2d/TensorArrayV2/element_shape:output:0"conv_lstm2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aconv_lstm2d/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          
3conv_lstm2d/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lstm2d/transpose:y:0Jconv_lstm2d/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
!conv_lstm2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv_lstm2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv_lstm2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
conv_lstm2d/strided_slice_1StridedSliceconv_lstm2d/transpose:y:0*conv_lstm2d/strided_slice_1/stack:output:0,conv_lstm2d/strided_slice_1/stack_1:output:0,conv_lstm2d/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_mask]
conv_lstm2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 conv_lstm2d/split/ReadVariableOpReadVariableOp)conv_lstm2d_split_readvariableop_resource*&
_output_shapes
: *
dtype0â
conv_lstm2d/splitSplit$conv_lstm2d/split/split_dim:output:0(conv_lstm2d/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split_
conv_lstm2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
"conv_lstm2d/split_1/ReadVariableOpReadVariableOp+conv_lstm2d_split_1_readvariableop_resource*&
_output_shapes
: *
dtype0è
conv_lstm2d/split_1Split&conv_lstm2d/split_1/split_dim:output:0*conv_lstm2d/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split_
conv_lstm2d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"conv_lstm2d/split_2/ReadVariableOpReadVariableOp+conv_lstm2d_split_2_readvariableop_resource*
_output_shapes
: *
dtype0¸
conv_lstm2d/split_2Split&conv_lstm2d/split_2/split_dim:output:0*conv_lstm2d/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÇ
conv_lstm2d/convolution_1Conv2D$conv_lstm2d/strided_slice_1:output:0conv_lstm2d/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/BiasAddBiasAdd"conv_lstm2d/convolution_1:output:0conv_lstm2d/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
conv_lstm2d/convolution_2Conv2D$conv_lstm2d/strided_slice_1:output:0conv_lstm2d/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/BiasAdd_1BiasAdd"conv_lstm2d/convolution_2:output:0conv_lstm2d/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
conv_lstm2d/convolution_3Conv2D$conv_lstm2d/strided_slice_1:output:0conv_lstm2d/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/BiasAdd_2BiasAdd"conv_lstm2d/convolution_3:output:0conv_lstm2d/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
conv_lstm2d/convolution_4Conv2D$conv_lstm2d/strided_slice_1:output:0conv_lstm2d/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/BiasAdd_3BiasAdd"conv_lstm2d/convolution_4:output:0conv_lstm2d/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Å
conv_lstm2d/convolution_5Conv2D conv_lstm2d/convolution:output:0conv_lstm2d/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Å
conv_lstm2d/convolution_6Conv2D conv_lstm2d/convolution:output:0conv_lstm2d/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Å
conv_lstm2d/convolution_7Conv2D conv_lstm2d/convolution:output:0conv_lstm2d/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Å
conv_lstm2d/convolution_8Conv2D conv_lstm2d/convolution:output:0conv_lstm2d/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/addAddV2conv_lstm2d/BiasAdd:output:0"conv_lstm2d/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
conv_lstm2d/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>X
conv_lstm2d/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/MulMulconv_lstm2d/add:z:0conv_lstm2d/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/Add_1AddV2conv_lstm2d/Mul:z:0conv_lstm2d/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ h
#conv_lstm2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
!conv_lstm2d/clip_by_value/MinimumMinimumconv_lstm2d/Add_1:z:0,conv_lstm2d/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ `
conv_lstm2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    «
conv_lstm2d/clip_by_valueMaximum%conv_lstm2d/clip_by_value/Minimum:z:0$conv_lstm2d/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/add_2AddV2conv_lstm2d/BiasAdd_1:output:0"conv_lstm2d/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ X
conv_lstm2d/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>X
conv_lstm2d/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/Mul_1Mulconv_lstm2d/add_2:z:0conv_lstm2d/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/Add_3AddV2conv_lstm2d/Mul_1:z:0conv_lstm2d/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
%conv_lstm2d/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
#conv_lstm2d/clip_by_value_1/MinimumMinimumconv_lstm2d/Add_3:z:0.conv_lstm2d/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
conv_lstm2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
conv_lstm2d/clip_by_value_1Maximum'conv_lstm2d/clip_by_value_1/Minimum:z:0&conv_lstm2d/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/mul_2Mulconv_lstm2d/clip_by_value_1:z:0 conv_lstm2d/convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/add_4AddV2conv_lstm2d/BiasAdd_2:output:0"conv_lstm2d/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ i
conv_lstm2d/ReluReluconv_lstm2d/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/mul_3Mulconv_lstm2d/clip_by_value:z:0conv_lstm2d/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/add_5AddV2conv_lstm2d/mul_2:z:0conv_lstm2d/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/add_6AddV2conv_lstm2d/BiasAdd_3:output:0"conv_lstm2d/convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ X
conv_lstm2d/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>X
conv_lstm2d/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/Mul_4Mulconv_lstm2d/add_6:z:0conv_lstm2d/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/Add_7AddV2conv_lstm2d/Mul_4:z:0conv_lstm2d/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
%conv_lstm2d/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
#conv_lstm2d/clip_by_value_2/MinimumMinimumconv_lstm2d/Add_7:z:0.conv_lstm2d/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
conv_lstm2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
conv_lstm2d/clip_by_value_2Maximum'conv_lstm2d/clip_by_value_2/Minimum:z:0&conv_lstm2d/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ k
conv_lstm2d/Relu_1Reluconv_lstm2d/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/mul_5Mulconv_lstm2d/clip_by_value_2:z:0 conv_lstm2d/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
)conv_lstm2d/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ú
conv_lstm2d/TensorArrayV2_1TensorListReserve2conv_lstm2d/TensorArrayV2_1/element_shape:output:0"conv_lstm2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒR
conv_lstm2d/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$conv_lstm2d/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ`
conv_lstm2d/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©
conv_lstm2d/whileWhile'conv_lstm2d/while/loop_counter:output:0-conv_lstm2d/while/maximum_iterations:output:0conv_lstm2d/time:output:0$conv_lstm2d/TensorArrayV2_1:handle:0 conv_lstm2d/convolution:output:0 conv_lstm2d/convolution:output:0"conv_lstm2d/strided_slice:output:0Cconv_lstm2d/TensorArrayUnstack/TensorListFromTensor:output_handle:0)conv_lstm2d_split_readvariableop_resource+conv_lstm2d_split_1_readvariableop_resource+conv_lstm2d_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
conv_lstm2d_while_body_171887*)
cond!R
conv_lstm2d_while_cond_171886*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
<conv_lstm2d/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          î
.conv_lstm2d/TensorArrayV2Stack/TensorListStackTensorListStackconv_lstm2d/while:output:3Econv_lstm2d/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ *
element_dtype0t
!conv_lstm2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#conv_lstm2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#conv_lstm2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
conv_lstm2d/strided_slice_2StridedSlice7conv_lstm2d/TensorArrayV2Stack/TensorListStack:tensor:0*conv_lstm2d/strided_slice_2/stack:output:0,conv_lstm2d/strided_slice_2/stack_1:output:0,conv_lstm2d/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_masky
conv_lstm2d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                Â
conv_lstm2d/transpose_1	Transpose7conv_lstm2d/TensorArrayV2Stack/TensorListStack:tensor:0%conv_lstm2d/transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          £
time_distributed/ReshapeReshapeconv_lstm2d/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Á
&time_distributed/max_pooling2d/MaxPoolMaxPool!time_distributed/Reshape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
}
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿd             ¿
time_distributed/Reshape_1Reshape/time_distributed/max_pooling2d/MaxPool:output:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          §
time_distributed/Reshape_2Reshapeconv_lstm2d/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d_1/zeros_like	ZerosLike#time_distributed/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd e
#conv_lstm2d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d_1/SumSumconv_lstm2d_1/zeros_like:y:0,conv_lstm2d_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
#conv_lstm2d_1/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            ^
conv_lstm2d_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
conv_lstm2d_1/zerosFill,conv_lstm2d_1/zeros/shape_as_tensor:output:0"conv_lstm2d_1/zeros/Const:output:0*
T0*&
_output_shapes
:¿
conv_lstm2d_1/convolutionConv2Dconv_lstm2d_1/Sum:output:0conv_lstm2d_1/zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
y
conv_lstm2d_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ®
conv_lstm2d_1/transpose	Transpose#time_distributed/Reshape_1:output:0%conv_lstm2d_1/transpose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ ^
conv_lstm2d_1/ShapeShapeconv_lstm2d_1/transpose:y:0*
T0*
_output_shapes
:k
!conv_lstm2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv_lstm2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv_lstm2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv_lstm2d_1/strided_sliceStridedSliceconv_lstm2d_1/Shape:output:0*conv_lstm2d_1/strided_slice/stack:output:0,conv_lstm2d_1/strided_slice/stack_1:output:0,conv_lstm2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)conv_lstm2d_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÜ
conv_lstm2d_1/TensorArrayV2TensorListReserve2conv_lstm2d_1/TensorArrayV2/element_shape:output:0$conv_lstm2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Cconv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          
5conv_lstm2d_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lstm2d_1/transpose:y:0Lconv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#conv_lstm2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%conv_lstm2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%conv_lstm2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:·
conv_lstm2d_1/strided_slice_1StridedSliceconv_lstm2d_1/transpose:y:0,conv_lstm2d_1/strided_slice_1/stack:output:0.conv_lstm2d_1/strided_slice_1/stack_1:output:0.conv_lstm2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask_
conv_lstm2d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
"conv_lstm2d_1/split/ReadVariableOpReadVariableOp+conv_lstm2d_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype0è
conv_lstm2d_1/splitSplit&conv_lstm2d_1/split/split_dim:output:0*conv_lstm2d_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splita
conv_lstm2d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$conv_lstm2d_1/split_1/ReadVariableOpReadVariableOp-conv_lstm2d_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype0î
conv_lstm2d_1/split_1Split(conv_lstm2d_1/split_1/split_dim:output:0,conv_lstm2d_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splita
conv_lstm2d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
$conv_lstm2d_1/split_2/ReadVariableOpReadVariableOp-conv_lstm2d_1_split_2_readvariableop_resource*
_output_shapes
:@*
dtype0¾
conv_lstm2d_1/split_2Split(conv_lstm2d_1/split_2/split_dim:output:0,conv_lstm2d_1/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÍ
conv_lstm2d_1/convolution_1Conv2D&conv_lstm2d_1/strided_slice_1:output:0conv_lstm2d_1/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
 
conv_lstm2d_1/BiasAddBiasAdd$conv_lstm2d_1/convolution_1:output:0conv_lstm2d_1/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Í
conv_lstm2d_1/convolution_2Conv2D&conv_lstm2d_1/strided_slice_1:output:0conv_lstm2d_1/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¢
conv_lstm2d_1/BiasAdd_1BiasAdd$conv_lstm2d_1/convolution_2:output:0conv_lstm2d_1/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Í
conv_lstm2d_1/convolution_3Conv2D&conv_lstm2d_1/strided_slice_1:output:0conv_lstm2d_1/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¢
conv_lstm2d_1/BiasAdd_2BiasAdd$conv_lstm2d_1/convolution_3:output:0conv_lstm2d_1/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Í
conv_lstm2d_1/convolution_4Conv2D&conv_lstm2d_1/strided_slice_1:output:0conv_lstm2d_1/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¢
conv_lstm2d_1/BiasAdd_3BiasAdd$conv_lstm2d_1/convolution_4:output:0conv_lstm2d_1/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ë
conv_lstm2d_1/convolution_5Conv2D"conv_lstm2d_1/convolution:output:0conv_lstm2d_1/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ë
conv_lstm2d_1/convolution_6Conv2D"conv_lstm2d_1/convolution:output:0conv_lstm2d_1/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ë
conv_lstm2d_1/convolution_7Conv2D"conv_lstm2d_1/convolution:output:0conv_lstm2d_1/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ë
conv_lstm2d_1/convolution_8Conv2D"conv_lstm2d_1/convolution:output:0conv_lstm2d_1/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv_lstm2d_1/addAddV2conv_lstm2d_1/BiasAdd:output:0$conv_lstm2d_1/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
conv_lstm2d_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Z
conv_lstm2d_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/MulMulconv_lstm2d_1/add:z:0conv_lstm2d_1/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/Add_1AddV2conv_lstm2d_1/Mul:z:0conv_lstm2d_1/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
%conv_lstm2d_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
#conv_lstm2d_1/clip_by_value/MinimumMinimumconv_lstm2d_1/Add_1:z:0.conv_lstm2d_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
conv_lstm2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
conv_lstm2d_1/clip_by_valueMaximum'conv_lstm2d_1/clip_by_value/Minimum:z:0&conv_lstm2d_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/add_2AddV2 conv_lstm2d_1/BiasAdd_1:output:0$conv_lstm2d_1/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
conv_lstm2d_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Z
conv_lstm2d_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/Mul_1Mulconv_lstm2d_1/add_2:z:0conv_lstm2d_1/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/Add_3AddV2conv_lstm2d_1/Mul_1:z:0conv_lstm2d_1/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
'conv_lstm2d_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
%conv_lstm2d_1/clip_by_value_1/MinimumMinimumconv_lstm2d_1/Add_3:z:00conv_lstm2d_1/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
conv_lstm2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
conv_lstm2d_1/clip_by_value_1Maximum)conv_lstm2d_1/clip_by_value_1/Minimum:z:0(conv_lstm2d_1/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/mul_2Mul!conv_lstm2d_1/clip_by_value_1:z:0"conv_lstm2d_1/convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/add_4AddV2 conv_lstm2d_1/BiasAdd_2:output:0$conv_lstm2d_1/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
conv_lstm2d_1/ReluReluconv_lstm2d_1/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/mul_3Mulconv_lstm2d_1/clip_by_value:z:0 conv_lstm2d_1/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/add_5AddV2conv_lstm2d_1/mul_2:z:0conv_lstm2d_1/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/add_6AddV2 conv_lstm2d_1/BiasAdd_3:output:0$conv_lstm2d_1/convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
conv_lstm2d_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Z
conv_lstm2d_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/Mul_4Mulconv_lstm2d_1/add_6:z:0conv_lstm2d_1/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/Add_7AddV2conv_lstm2d_1/Mul_4:z:0conv_lstm2d_1/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
'conv_lstm2d_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
%conv_lstm2d_1/clip_by_value_2/MinimumMinimumconv_lstm2d_1/Add_7:z:00conv_lstm2d_1/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
conv_lstm2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
conv_lstm2d_1/clip_by_value_2Maximum)conv_lstm2d_1/clip_by_value_2/Minimum:z:0(conv_lstm2d_1/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
conv_lstm2d_1/Relu_1Reluconv_lstm2d_1/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/mul_5Mul!conv_lstm2d_1/clip_by_value_2:z:0"conv_lstm2d_1/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+conv_lstm2d_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
conv_lstm2d_1/TensorArrayV2_1TensorListReserve4conv_lstm2d_1/TensorArrayV2_1/element_shape:output:0$conv_lstm2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
conv_lstm2d_1/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&conv_lstm2d_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 conv_lstm2d_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Å
conv_lstm2d_1/whileWhile)conv_lstm2d_1/while/loop_counter:output:0/conv_lstm2d_1/while/maximum_iterations:output:0conv_lstm2d_1/time:output:0&conv_lstm2d_1/TensorArrayV2_1:handle:0"conv_lstm2d_1/convolution:output:0"conv_lstm2d_1/convolution:output:0$conv_lstm2d_1/strided_slice:output:0Econv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0+conv_lstm2d_1_split_readvariableop_resource-conv_lstm2d_1_split_1_readvariableop_resource-conv_lstm2d_1_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
conv_lstm2d_1_while_body_172112*+
cond#R!
conv_lstm2d_1_while_cond_172111*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
>conv_lstm2d_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ô
0conv_lstm2d_1/TensorArrayV2Stack/TensorListStackTensorListStackconv_lstm2d_1/while:output:3Gconv_lstm2d_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ *
element_dtype0v
#conv_lstm2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%conv_lstm2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%conv_lstm2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Õ
conv_lstm2d_1/strided_slice_2StridedSlice9conv_lstm2d_1/TensorArrayV2Stack/TensorListStack:tensor:0,conv_lstm2d_1/strided_slice_2/stack:output:0.conv_lstm2d_1/strided_slice_2/stack_1:output:0.conv_lstm2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask{
conv_lstm2d_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                È
conv_lstm2d_1/transpose_1	Transpose9conv_lstm2d_1/TensorArrayV2Stack/TensorListStack:tensor:0'conv_lstm2d_1/transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ©
time_distributed_1/ReshapeReshapeconv_lstm2d_1/transpose_1:y:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
&time_distributed_1/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
(time_distributed_1/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ´
$time_distributed_1/up_sampling2d/mulMul/time_distributed_1/up_sampling2d/Const:output:01time_distributed_1/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:ù
=time_distributed_1/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#time_distributed_1/Reshape:output:0(time_distributed_1/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
half_pixel_centers(
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿd   @          â
time_distributed_1/Reshape_1ReshapeNtime_distributed_1/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ {
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ­
time_distributed_1/Reshape_2Reshapeconv_lstm2d_1/transpose_1:y:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
conv2d/Conv2D/ShapeShape%time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿm
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          §
conv2d/Conv2D/ReshapeReshape%time_distributed_1/Reshape_1:output:0$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Í
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿy
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          º
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¦
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ì
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Ã
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ ^
conv2d_1/Conv2D/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          
conv2d_1/Conv2D/ReshapeReshapeconv2d/Relu:activations:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0Ó
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@       
   À
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
ª
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ò
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 

+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 

conv2d_1/SoftmaxSoftmax.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
u
IdentityIdentityconv2d_1/Softmax:softmax:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 

NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp!^conv_lstm2d/split/ReadVariableOp#^conv_lstm2d/split_1/ReadVariableOp#^conv_lstm2d/split_2/ReadVariableOp^conv_lstm2d/while#^conv_lstm2d_1/split/ReadVariableOp%^conv_lstm2d_1/split_1/ReadVariableOp%^conv_lstm2d_1/split_2/ReadVariableOp^conv_lstm2d_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2D
 conv_lstm2d/split/ReadVariableOp conv_lstm2d/split/ReadVariableOp2H
"conv_lstm2d/split_1/ReadVariableOp"conv_lstm2d/split_1/ReadVariableOp2H
"conv_lstm2d/split_2/ReadVariableOp"conv_lstm2d/split_2/ReadVariableOp2&
conv_lstm2d/whileconv_lstm2d/while2H
"conv_lstm2d_1/split/ReadVariableOp"conv_lstm2d_1/split/ReadVariableOp2L
$conv_lstm2d_1/split_1/ReadVariableOp$conv_lstm2d_1/split_1/ReadVariableOp2L
$conv_lstm2d_1/split_2/ReadVariableOp$conv_lstm2d_1/split_2/ReadVariableOp2*
conv_lstm2d_1/whileconv_lstm2d_1/while:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
³1
é
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_169187

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
identity¢StatefulPartitionedCall¢whilef

zeros_like	ZerosLikeinputs*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ~
	transpose	Transposeinputstranspose/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskï
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_169105v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_169119*
condR
while_cond_169118*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ó
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                §
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ s
IdentityIdentitytranspose_1:y:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ h
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ : : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs
øb
Ü
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_172597
inputs_07
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢whileh

zeros_like	ZerosLikeinputs_0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_172471*
condR
while_cond_172470*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ó
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                §
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ s
IdentityIdentitytranspose_1:y:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
inputs/0
Ñ
Á
while_cond_173442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_173442___redundant_placeholder04
0while_while_cond_173442___redundant_placeholder14
0while_while_cond_173442___redundant_placeholder24
0while_while_cond_173442___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ñ
Á
while_cond_169343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_169343___redundant_placeholder04
0while_while_cond_169343___redundant_placeholder14
0while_while_cond_169343___redundant_placeholder24
0while_while_cond_169343___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
Ñ
¼
"__inference__traced_restore_175033
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:
.
 assignvariableop_3_conv2d_1_bias:
&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ?
%assignvariableop_9_conv_lstm2d_kernel: J
0assignvariableop_10_conv_lstm2d_recurrent_kernel: 2
$assignvariableop_11_conv_lstm2d_bias: B
(assignvariableop_12_conv_lstm2d_1_kernel:@L
2assignvariableop_13_conv_lstm2d_1_recurrent_kernel:@4
&assignvariableop_14_conv_lstm2d_1_bias:@#
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: B
(assignvariableop_19_adam_conv2d_kernel_m:4
&assignvariableop_20_adam_conv2d_bias_m:D
*assignvariableop_21_adam_conv2d_1_kernel_m:
6
(assignvariableop_22_adam_conv2d_1_bias_m:
G
-assignvariableop_23_adam_conv_lstm2d_kernel_m: Q
7assignvariableop_24_adam_conv_lstm2d_recurrent_kernel_m: 9
+assignvariableop_25_adam_conv_lstm2d_bias_m: I
/assignvariableop_26_adam_conv_lstm2d_1_kernel_m:@S
9assignvariableop_27_adam_conv_lstm2d_1_recurrent_kernel_m:@;
-assignvariableop_28_adam_conv_lstm2d_1_bias_m:@B
(assignvariableop_29_adam_conv2d_kernel_v:4
&assignvariableop_30_adam_conv2d_bias_v:D
*assignvariableop_31_adam_conv2d_1_kernel_v:
6
(assignvariableop_32_adam_conv2d_1_bias_v:
G
-assignvariableop_33_adam_conv_lstm2d_kernel_v: Q
7assignvariableop_34_adam_conv_lstm2d_recurrent_kernel_v: 9
+assignvariableop_35_adam_conv_lstm2d_bias_v: I
/assignvariableop_36_adam_conv_lstm2d_1_kernel_v:@S
9assignvariableop_37_adam_conv_lstm2d_1_recurrent_kernel_v:@;
-assignvariableop_38_adam_conv_lstm2d_1_bias_v:@
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Þ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueúB÷(B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B é
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_conv_lstm2d_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_10AssignVariableOp0assignvariableop_10_conv_lstm2d_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv_lstm2d_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp(assignvariableop_12_conv_lstm2d_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_13AssignVariableOp2assignvariableop_13_conv_lstm2d_1_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp&assignvariableop_14_conv_lstm2d_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_conv_lstm2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_24AssignVariableOp7assignvariableop_24_adam_conv_lstm2d_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv_lstm2d_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_26AssignVariableOp/assignvariableop_26_adam_conv_lstm2d_1_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_27AssignVariableOp9assignvariableop_27_adam_conv_lstm2d_1_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp-assignvariableop_28_adam_conv_lstm2d_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv2d_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp-assignvariableop_33_adam_conv_lstm2d_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_conv_lstm2d_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv_lstm2d_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_36AssignVariableOp/assignvariableop_36_adam_conv_lstm2d_1_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_37AssignVariableOp9assignvariableop_37_adam_conv_lstm2d_1_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp-assignvariableop_38_adam_conv_lstm2d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ©
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
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
Ê
©
#model_conv_lstm2d_while_cond_168583@
<model_conv_lstm2d_while_model_conv_lstm2d_while_loop_counterF
Bmodel_conv_lstm2d_while_model_conv_lstm2d_while_maximum_iterations'
#model_conv_lstm2d_while_placeholder)
%model_conv_lstm2d_while_placeholder_1)
%model_conv_lstm2d_while_placeholder_2)
%model_conv_lstm2d_while_placeholder_3@
<model_conv_lstm2d_while_less_model_conv_lstm2d_strided_sliceX
Tmodel_conv_lstm2d_while_model_conv_lstm2d_while_cond_168583___redundant_placeholder0X
Tmodel_conv_lstm2d_while_model_conv_lstm2d_while_cond_168583___redundant_placeholder1X
Tmodel_conv_lstm2d_while_model_conv_lstm2d_while_cond_168583___redundant_placeholder2X
Tmodel_conv_lstm2d_while_model_conv_lstm2d_while_cond_168583___redundant_placeholder3$
 model_conv_lstm2d_while_identity
¨
model/conv_lstm2d/while/LessLess#model_conv_lstm2d_while_placeholder<model_conv_lstm2d_while_less_model_conv_lstm2d_strided_slice*
T0*
_output_shapes
: o
 model/conv_lstm2d/while/IdentityIdentity model/conv_lstm2d/while/Less:z:0*
T0
*
_output_shapes
: "M
 model_conv_lstm2d_while_identity)model/conv_lstm2d/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
Z
ë
while_body_170302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0:@A
'while_split_1_readvariableop_resource_0:@5
'while_split_2_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource:@?
%while_split_1_readvariableop_resource:@3
%while_split_2_readvariableop_resource:@¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

M
1__inference_time_distributed_layer_call_fn_173267

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_169478u
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs
Ó=

L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_174674

inputs
states_0
states_17
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity

identity_1

identity_2¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
ý

)__inference_conv2d_1_layer_call_fn_174338

inputs!
unknown:

	unknown_0:

identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_170511{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
ß

/__inference_conv_lstm_cell_layer_call_fn_174388

inputs
states_0
states_1!
unknown: #
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_169105w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
states/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
states/1
¡%
¯
B__inference_conv2d_layer_call_and_return_conditional_losses_174329

inputs?
%conv2d_conv2d_readvariableop_resource:@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢Conv2D/Conv2D/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¸
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¥
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0·
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:®
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd@ : : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
Ñ=

J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_174555

inputs
states_0
states_17
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity

identity_1

identity_2¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ `
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
states/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
states/1
 $

A__inference_model_layer_call_and_return_conditional_losses_171108

inputs,
conv_lstm2d_171077: ,
conv_lstm2d_171079:  
conv_lstm2d_171081: .
conv_lstm2d_1_171087:@.
conv_lstm2d_1_171089:@"
conv_lstm2d_1_171091:@'
conv2d_171097:
conv2d_171099:)
conv2d_1_171102:

conv2d_1_171104:

identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢#conv_lstm2d/StatefulPartitionedCall¢%conv_lstm2d_1/StatefulPartitionedCall¡
#conv_lstm2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv_lstm2d_171077conv_lstm2d_171079conv_lstm2d_171081*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_171038ý
 time_distributed/PartitionedCallPartitionedCall,conv_lstm2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_169478w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ´
time_distributed/ReshapeReshape,conv_lstm2d/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Î
%conv_lstm2d_1/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0conv_lstm2d_1_171087conv_lstm2d_1_171089conv_lstm2d_1_171091*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_170796
"time_distributed_1/PartitionedCallPartitionedCall.conv_lstm2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_169967y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          º
time_distributed_1/ReshapeReshape.conv_lstm2d_1/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:0conv2d_171097conv2d_171099*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_170472 
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_171102conv2d_1_171104*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_170511
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
Ø
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall$^conv_lstm2d/StatefulPartitionedCall&^conv_lstm2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2J
#conv_lstm2d/StatefulPartitionedCall#conv_lstm2d/StatefulPartitionedCall2N
%conv_lstm2d_1/StatefulPartitionedCall%conv_lstm2d_1/StatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs

j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_169944

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ é
up_sampling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_169919\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&up_sampling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_174565

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£$

A__inference_model_layer_call_and_return_conditional_losses_171190
input_1,
conv_lstm2d_171159: ,
conv_lstm2d_171161:  
conv_lstm2d_171163: .
conv_lstm2d_1_171169:@.
conv_lstm2d_1_171171:@"
conv_lstm2d_1_171173:@'
conv2d_171179:
conv2d_171181:)
conv2d_1_171184:

conv2d_1_171186:

identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢#conv_lstm2d/StatefulPartitionedCall¢%conv_lstm2d_1/StatefulPartitionedCall¢
#conv_lstm2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_lstm2d_171159conv_lstm2d_171161conv_lstm2d_171163*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_170196ý
 time_distributed/PartitionedCallPartitionedCall,conv_lstm2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_169455w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ´
time_distributed/ReshapeReshape,conv_lstm2d/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Î
%conv_lstm2d_1/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0conv_lstm2d_1_171169conv_lstm2d_1_171171conv_lstm2d_1_171173*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_170428
"time_distributed_1/PartitionedCallPartitionedCall.conv_lstm2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_169944y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          º
time_distributed_1/ReshapeReshape.conv_lstm2d_1/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:0conv2d_171179conv2d_171181*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_170472 
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_171184conv2d_1_171186*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_170511
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
Ø
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall$^conv_lstm2d/StatefulPartitionedCall&^conv_lstm2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2J
#conv_lstm2d/StatefulPartitionedCall#conv_lstm2d/StatefulPartitionedCall2N
%conv_lstm2d_1/StatefulPartitionedCall%conv_lstm2d_1/StatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
!
_user_specified_name	input_1
¿!

while_body_169826
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_169850_0:@(
while_169852_0:@
while_169854_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_169850:@&
while_169852:@
while_169854:@¢while/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_169850_0while_169852_0while_169854_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_169773Ï
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
while_169850while_169850_0"
while_169852while_169852_0"
while_169854while_169854_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
t
«
conv_lstm2d_1_while_body_1715998
4conv_lstm2d_1_while_conv_lstm2d_1_while_loop_counter>
:conv_lstm2d_1_while_conv_lstm2d_1_while_maximum_iterations#
conv_lstm2d_1_while_placeholder%
!conv_lstm2d_1_while_placeholder_1%
!conv_lstm2d_1_while_placeholder_2%
!conv_lstm2d_1_while_placeholder_35
1conv_lstm2d_1_while_conv_lstm2d_1_strided_slice_0s
oconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0M
3conv_lstm2d_1_while_split_readvariableop_resource_0:@O
5conv_lstm2d_1_while_split_1_readvariableop_resource_0:@C
5conv_lstm2d_1_while_split_2_readvariableop_resource_0:@ 
conv_lstm2d_1_while_identity"
conv_lstm2d_1_while_identity_1"
conv_lstm2d_1_while_identity_2"
conv_lstm2d_1_while_identity_3"
conv_lstm2d_1_while_identity_4"
conv_lstm2d_1_while_identity_53
/conv_lstm2d_1_while_conv_lstm2d_1_strided_sliceq
mconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensorK
1conv_lstm2d_1_while_split_readvariableop_resource:@M
3conv_lstm2d_1_while_split_1_readvariableop_resource:@A
3conv_lstm2d_1_while_split_2_readvariableop_resource:@¢(conv_lstm2d_1/while/split/ReadVariableOp¢*conv_lstm2d_1/while/split_1/ReadVariableOp¢*conv_lstm2d_1/while/split_2/ReadVariableOp
Econv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ô
7conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0conv_lstm2d_1_while_placeholderNconv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0e
#conv_lstm2d_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¤
(conv_lstm2d_1/while/split/ReadVariableOpReadVariableOp3conv_lstm2d_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0ú
conv_lstm2d_1/while/splitSplit,conv_lstm2d_1/while/split/split_dim:output:00conv_lstm2d_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitg
%conv_lstm2d_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
*conv_lstm2d_1/while/split_1/ReadVariableOpReadVariableOp5conv_lstm2d_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0
conv_lstm2d_1/while/split_1Split.conv_lstm2d_1/while/split_1/split_dim:output:02conv_lstm2d_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitg
%conv_lstm2d_1/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
*conv_lstm2d_1/while/split_2/ReadVariableOpReadVariableOp5conv_lstm2d_1_while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ð
conv_lstm2d_1/while/split_2Split.conv_lstm2d_1/while/split_2/split_dim:output:02conv_lstm2d_1/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitï
conv_lstm2d_1/while/convolutionConv2D>conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_1/while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
°
conv_lstm2d_1/while/BiasAddBiasAdd(conv_lstm2d_1/while/convolution:output:0$conv_lstm2d_1/while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ñ
!conv_lstm2d_1/while/convolution_1Conv2D>conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_1/while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
conv_lstm2d_1/while/BiasAdd_1BiasAdd*conv_lstm2d_1/while/convolution_1:output:0$conv_lstm2d_1/while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ñ
!conv_lstm2d_1/while/convolution_2Conv2D>conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_1/while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
conv_lstm2d_1/while/BiasAdd_2BiasAdd*conv_lstm2d_1/while/convolution_2:output:0$conv_lstm2d_1/while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ñ
!conv_lstm2d_1/while/convolution_3Conv2D>conv_lstm2d_1/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_1/while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
conv_lstm2d_1/while/BiasAdd_3BiasAdd*conv_lstm2d_1/while/convolution_3:output:0$conv_lstm2d_1/while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
!conv_lstm2d_1/while/convolution_4Conv2D!conv_lstm2d_1_while_placeholder_2$conv_lstm2d_1/while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ö
!conv_lstm2d_1/while/convolution_5Conv2D!conv_lstm2d_1_while_placeholder_2$conv_lstm2d_1/while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ö
!conv_lstm2d_1/while/convolution_6Conv2D!conv_lstm2d_1_while_placeholder_2$conv_lstm2d_1/while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ö
!conv_lstm2d_1/while/convolution_7Conv2D!conv_lstm2d_1_while_placeholder_2$conv_lstm2d_1/while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
conv_lstm2d_1/while/addAddV2$conv_lstm2d_1/while/BiasAdd:output:0*conv_lstm2d_1/while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
conv_lstm2d_1/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
conv_lstm2d_1/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/while/MulMulconv_lstm2d_1/while/add:z:0"conv_lstm2d_1/while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/while/Add_1AddV2conv_lstm2d_1/while/Mul:z:0$conv_lstm2d_1/while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
+conv_lstm2d_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
)conv_lstm2d_1/while/clip_by_value/MinimumMinimumconv_lstm2d_1/while/Add_1:z:04conv_lstm2d_1/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
#conv_lstm2d_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!conv_lstm2d_1/while/clip_by_valueMaximum-conv_lstm2d_1/while/clip_by_value/Minimum:z:0,conv_lstm2d_1/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
conv_lstm2d_1/while/add_2AddV2&conv_lstm2d_1/while/BiasAdd_1:output:0*conv_lstm2d_1/while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
conv_lstm2d_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
conv_lstm2d_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/while/Mul_1Mulconv_lstm2d_1/while/add_2:z:0$conv_lstm2d_1/while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
conv_lstm2d_1/while/Add_3AddV2conv_lstm2d_1/while/Mul_1:z:0$conv_lstm2d_1/while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
-conv_lstm2d_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
+conv_lstm2d_1/while/clip_by_value_1/MinimumMinimumconv_lstm2d_1/while/Add_3:z:06conv_lstm2d_1/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
%conv_lstm2d_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    É
#conv_lstm2d_1/while/clip_by_value_1Maximum/conv_lstm2d_1/while/clip_by_value_1/Minimum:z:0.conv_lstm2d_1/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
conv_lstm2d_1/while/mul_2Mul'conv_lstm2d_1/while/clip_by_value_1:z:0!conv_lstm2d_1_while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
conv_lstm2d_1/while/add_4AddV2&conv_lstm2d_1/while/BiasAdd_2:output:0*conv_lstm2d_1/while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
conv_lstm2d_1/while/ReluReluconv_lstm2d_1/while/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
conv_lstm2d_1/while/mul_3Mul%conv_lstm2d_1/while/clip_by_value:z:0&conv_lstm2d_1/while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/while/add_5AddV2conv_lstm2d_1/while/mul_2:z:0conv_lstm2d_1/while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
conv_lstm2d_1/while/add_6AddV2&conv_lstm2d_1/while/BiasAdd_3:output:0*conv_lstm2d_1/while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
conv_lstm2d_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
conv_lstm2d_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/while/Mul_4Mulconv_lstm2d_1/while/add_6:z:0$conv_lstm2d_1/while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
conv_lstm2d_1/while/Add_7AddV2conv_lstm2d_1/while/Mul_4:z:0$conv_lstm2d_1/while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
-conv_lstm2d_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
+conv_lstm2d_1/while/clip_by_value_2/MinimumMinimumconv_lstm2d_1/while/Add_7:z:06conv_lstm2d_1/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
%conv_lstm2d_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    É
#conv_lstm2d_1/while/clip_by_value_2Maximum/conv_lstm2d_1/while/clip_by_value_2/Minimum:z:0.conv_lstm2d_1/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
conv_lstm2d_1/while/Relu_1Reluconv_lstm2d_1/while/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
conv_lstm2d_1/while/mul_5Mul'conv_lstm2d_1/while/clip_by_value_2:z:0(conv_lstm2d_1/while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
8conv_lstm2d_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!conv_lstm2d_1_while_placeholder_1conv_lstm2d_1_while_placeholderconv_lstm2d_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒ]
conv_lstm2d_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d_1/while/add_8AddV2conv_lstm2d_1_while_placeholder$conv_lstm2d_1/while/add_8/y:output:0*
T0*
_output_shapes
: ]
conv_lstm2d_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d_1/while/add_9AddV24conv_lstm2d_1_while_conv_lstm2d_1_while_loop_counter$conv_lstm2d_1/while/add_9/y:output:0*
T0*
_output_shapes
: 
conv_lstm2d_1/while/IdentityIdentityconv_lstm2d_1/while/add_9:z:0^conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: ¢
conv_lstm2d_1/while/Identity_1Identity:conv_lstm2d_1_while_conv_lstm2d_1_while_maximum_iterations^conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: 
conv_lstm2d_1/while/Identity_2Identityconv_lstm2d_1/while/add_8:z:0^conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: Ã
conv_lstm2d_1/while/Identity_3IdentityHconv_lstm2d_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lstm2d_1/while/NoOp*
T0*
_output_shapes
: :éèÒ
conv_lstm2d_1/while/Identity_4Identityconv_lstm2d_1/while/mul_5:z:0^conv_lstm2d_1/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/while/Identity_5Identityconv_lstm2d_1/while/add_5:z:0^conv_lstm2d_1/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß
conv_lstm2d_1/while/NoOpNoOp)^conv_lstm2d_1/while/split/ReadVariableOp+^conv_lstm2d_1/while/split_1/ReadVariableOp+^conv_lstm2d_1/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/conv_lstm2d_1_while_conv_lstm2d_1_strided_slice1conv_lstm2d_1_while_conv_lstm2d_1_strided_slice_0"E
conv_lstm2d_1_while_identity%conv_lstm2d_1/while/Identity:output:0"I
conv_lstm2d_1_while_identity_1'conv_lstm2d_1/while/Identity_1:output:0"I
conv_lstm2d_1_while_identity_2'conv_lstm2d_1/while/Identity_2:output:0"I
conv_lstm2d_1_while_identity_3'conv_lstm2d_1/while/Identity_3:output:0"I
conv_lstm2d_1_while_identity_4'conv_lstm2d_1/while/Identity_4:output:0"I
conv_lstm2d_1_while_identity_5'conv_lstm2d_1/while/Identity_5:output:0"l
3conv_lstm2d_1_while_split_1_readvariableop_resource5conv_lstm2d_1_while_split_1_readvariableop_resource_0"l
3conv_lstm2d_1_while_split_2_readvariableop_resource5conv_lstm2d_1_while_split_2_readvariableop_resource_0"h
1conv_lstm2d_1_while_split_readvariableop_resource3conv_lstm2d_1_while_split_readvariableop_resource_0"à
mconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensoroconv_lstm2d_1_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2T
(conv_lstm2d_1/while/split/ReadVariableOp(conv_lstm2d_1/while/split/ReadVariableOp2X
*conv_lstm2d_1/while/split_1/ReadVariableOp*conv_lstm2d_1/while/split_1/ReadVariableOp2X
*conv_lstm2d_1/while/split_2/ReadVariableOp*conv_lstm2d_1/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

O
3__inference_time_distributed_1_layer_call_fn_174240

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_169944u
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

M
1__inference_time_distributed_layer_call_fn_173262

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_169455u
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs
·
J
.__inference_up_sampling2d_layer_call_fn_174754

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_169919
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°b
Ú
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_173257

inputs7
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢while]

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                u
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_173131*
condR
while_cond_173130*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ê
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ j
IdentityIdentitytranspose_1:y:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
Ñ
Á
while_cond_169598
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_169598___redundant_placeholder04
0while_while_cond_169598___redundant_placeholder14
0while_while_cond_169598___redundant_placeholder24
0while_while_cond_169598___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Z
ë
while_body_173443
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0:@A
'while_split_1_readvariableop_resource_0:@5
'while_split_2_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource:@?
%while_split_1_readvariableop_resource:@3
%while_split_2_readvariableop_resource:@¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
øb
Ü
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_172817
inputs_07
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢whileh

zeros_like	ZerosLikeinputs_0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_172691*
condR
while_cond_172690*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ó
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                §
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ s
IdentityIdentitytranspose_1:y:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
inputs/0
Áï
¿

!__inference__wrapped_model_169003
input_1I
/model_conv_lstm2d_split_readvariableop_resource: K
1model_conv_lstm2d_split_1_readvariableop_resource: ?
1model_conv_lstm2d_split_2_readvariableop_resource: K
1model_conv_lstm2d_1_split_readvariableop_resource:@M
3model_conv_lstm2d_1_split_1_readvariableop_resource:@A
3model_conv_lstm2d_1_split_2_readvariableop_resource:@L
2model_conv2d_conv2d_conv2d_readvariableop_resource:M
?model_conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:N
4model_conv2d_1_conv2d_conv2d_readvariableop_resource:
O
Amodel_conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:

identity¢)model/conv2d/Conv2D/Conv2D/ReadVariableOp¢6model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp¢+model/conv2d_1/Conv2D/Conv2D/ReadVariableOp¢8model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp¢&model/conv_lstm2d/split/ReadVariableOp¢(model/conv_lstm2d/split_1/ReadVariableOp¢(model/conv_lstm2d/split_2/ReadVariableOp¢model/conv_lstm2d/while¢(model/conv_lstm2d_1/split/ReadVariableOp¢*model/conv_lstm2d_1/split_1/ReadVariableOp¢*model/conv_lstm2d_1/split_2/ReadVariableOp¢model/conv_lstm2d_1/whilep
model/conv_lstm2d/zeros_like	ZerosLikeinput_1*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ i
'model/conv_lstm2d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :ª
model/conv_lstm2d/SumSum model/conv_lstm2d/zeros_like:y:00model/conv_lstm2d/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ |
model/conv_lstm2d/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    Ë
model/conv_lstm2d/convolutionConv2Dmodel/conv_lstm2d/Sum:output:0 model/conv_lstm2d/zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
}
 model/conv_lstm2d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
model/conv_lstm2d/transpose	Transposeinput_1)model/conv_lstm2d/transpose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ f
model/conv_lstm2d/ShapeShapemodel/conv_lstm2d/transpose:y:0*
T0*
_output_shapes
:o
%model/conv_lstm2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model/conv_lstm2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model/conv_lstm2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
model/conv_lstm2d/strided_sliceStridedSlice model/conv_lstm2d/Shape:output:0.model/conv_lstm2d/strided_slice/stack:output:00model/conv_lstm2d/strided_slice/stack_1:output:00model/conv_lstm2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-model/conv_lstm2d/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
model/conv_lstm2d/TensorArrayV2TensorListReserve6model/conv_lstm2d/TensorArrayV2/element_shape:output:0(model/conv_lstm2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ 
Gmodel/conv_lstm2d/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          
9model/conv_lstm2d/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/conv_lstm2d/transpose:y:0Pmodel/conv_lstm2d/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒq
'model/conv_lstm2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)model/conv_lstm2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)model/conv_lstm2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
!model/conv_lstm2d/strided_slice_1StridedSlicemodel/conv_lstm2d/transpose:y:00model/conv_lstm2d/strided_slice_1/stack:output:02model/conv_lstm2d/strided_slice_1/stack_1:output:02model/conv_lstm2d/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskc
!model/conv_lstm2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&model/conv_lstm2d/split/ReadVariableOpReadVariableOp/model_conv_lstm2d_split_readvariableop_resource*&
_output_shapes
: *
dtype0ô
model/conv_lstm2d/splitSplit*model/conv_lstm2d/split/split_dim:output:0.model/conv_lstm2d/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splite
#model/conv_lstm2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¢
(model/conv_lstm2d/split_1/ReadVariableOpReadVariableOp1model_conv_lstm2d_split_1_readvariableop_resource*&
_output_shapes
: *
dtype0ú
model/conv_lstm2d/split_1Split,model/conv_lstm2d/split_1/split_dim:output:00model/conv_lstm2d/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splite
#model/conv_lstm2d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(model/conv_lstm2d/split_2/ReadVariableOpReadVariableOp1model_conv_lstm2d_split_2_readvariableop_resource*
_output_shapes
: *
dtype0Ê
model/conv_lstm2d/split_2Split,model/conv_lstm2d/split_2/split_dim:output:00model/conv_lstm2d/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÙ
model/conv_lstm2d/convolution_1Conv2D*model/conv_lstm2d/strided_slice_1:output:0 model/conv_lstm2d/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
model/conv_lstm2d/BiasAddBiasAdd(model/conv_lstm2d/convolution_1:output:0"model/conv_lstm2d/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ù
model/conv_lstm2d/convolution_2Conv2D*model/conv_lstm2d/strided_slice_1:output:0 model/conv_lstm2d/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
model/conv_lstm2d/BiasAdd_1BiasAdd(model/conv_lstm2d/convolution_2:output:0"model/conv_lstm2d/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ù
model/conv_lstm2d/convolution_3Conv2D*model/conv_lstm2d/strided_slice_1:output:0 model/conv_lstm2d/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
model/conv_lstm2d/BiasAdd_2BiasAdd(model/conv_lstm2d/convolution_3:output:0"model/conv_lstm2d/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ù
model/conv_lstm2d/convolution_4Conv2D*model/conv_lstm2d/strided_slice_1:output:0 model/conv_lstm2d/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
model/conv_lstm2d/BiasAdd_3BiasAdd(model/conv_lstm2d/convolution_4:output:0"model/conv_lstm2d/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ×
model/conv_lstm2d/convolution_5Conv2D&model/conv_lstm2d/convolution:output:0"model/conv_lstm2d/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
×
model/conv_lstm2d/convolution_6Conv2D&model/conv_lstm2d/convolution:output:0"model/conv_lstm2d/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
×
model/conv_lstm2d/convolution_7Conv2D&model/conv_lstm2d/convolution:output:0"model/conv_lstm2d/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
×
model/conv_lstm2d/convolution_8Conv2D&model/conv_lstm2d/convolution:output:0"model/conv_lstm2d/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¦
model/conv_lstm2d/addAddV2"model/conv_lstm2d/BiasAdd:output:0(model/conv_lstm2d/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
model/conv_lstm2d/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
model/conv_lstm2d/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
model/conv_lstm2d/MulMulmodel/conv_lstm2d/add:z:0 model/conv_lstm2d/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
model/conv_lstm2d/Add_1AddV2model/conv_lstm2d/Mul:z:0"model/conv_lstm2d/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ n
)model/conv_lstm2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?½
'model/conv_lstm2d/clip_by_value/MinimumMinimummodel/conv_lstm2d/Add_1:z:02model/conv_lstm2d/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ f
!model/conv_lstm2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
model/conv_lstm2d/clip_by_valueMaximum+model/conv_lstm2d/clip_by_value/Minimum:z:0*model/conv_lstm2d/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
model/conv_lstm2d/add_2AddV2$model/conv_lstm2d/BiasAdd_1:output:0(model/conv_lstm2d/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
model/conv_lstm2d/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
model/conv_lstm2d/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
model/conv_lstm2d/Mul_1Mulmodel/conv_lstm2d/add_2:z:0"model/conv_lstm2d/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
model/conv_lstm2d/Add_3AddV2model/conv_lstm2d/Mul_1:z:0"model/conv_lstm2d/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
+model/conv_lstm2d/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
)model/conv_lstm2d/clip_by_value_1/MinimumMinimummodel/conv_lstm2d/Add_3:z:04model/conv_lstm2d/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ h
#model/conv_lstm2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!model/conv_lstm2d/clip_by_value_1Maximum-model/conv_lstm2d/clip_by_value_1/Minimum:z:0,model/conv_lstm2d/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §
model/conv_lstm2d/mul_2Mul%model/conv_lstm2d/clip_by_value_1:z:0&model/conv_lstm2d/convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
model/conv_lstm2d/add_4AddV2$model/conv_lstm2d/BiasAdd_2:output:0(model/conv_lstm2d/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
model/conv_lstm2d/ReluRelumodel/conv_lstm2d/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
model/conv_lstm2d/mul_3Mul#model/conv_lstm2d/clip_by_value:z:0$model/conv_lstm2d/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
model/conv_lstm2d/add_5AddV2model/conv_lstm2d/mul_2:z:0model/conv_lstm2d/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
model/conv_lstm2d/add_6AddV2$model/conv_lstm2d/BiasAdd_3:output:0(model/conv_lstm2d/convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
model/conv_lstm2d/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
model/conv_lstm2d/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
model/conv_lstm2d/Mul_4Mulmodel/conv_lstm2d/add_6:z:0"model/conv_lstm2d/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
model/conv_lstm2d/Add_7AddV2model/conv_lstm2d/Mul_4:z:0"model/conv_lstm2d/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
+model/conv_lstm2d/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
)model/conv_lstm2d/clip_by_value_2/MinimumMinimummodel/conv_lstm2d/Add_7:z:04model/conv_lstm2d/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ h
#model/conv_lstm2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!model/conv_lstm2d/clip_by_value_2Maximum-model/conv_lstm2d/clip_by_value_2/Minimum:z:0,model/conv_lstm2d/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
model/conv_lstm2d/Relu_1Relumodel/conv_lstm2d/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §
model/conv_lstm2d/mul_5Mul%model/conv_lstm2d/clip_by_value_2:z:0&model/conv_lstm2d/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
/model/conv_lstm2d/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ì
!model/conv_lstm2d/TensorArrayV2_1TensorListReserve8model/conv_lstm2d/TensorArrayV2_1/element_shape:output:0(model/conv_lstm2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
model/conv_lstm2d/timeConst*
_output_shapes
: *
dtype0*
value	B : u
*model/conv_lstm2d/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿf
$model/conv_lstm2d/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
model/conv_lstm2d/whileWhile-model/conv_lstm2d/while/loop_counter:output:03model/conv_lstm2d/while/maximum_iterations:output:0model/conv_lstm2d/time:output:0*model/conv_lstm2d/TensorArrayV2_1:handle:0&model/conv_lstm2d/convolution:output:0&model/conv_lstm2d/convolution:output:0(model/conv_lstm2d/strided_slice:output:0Imodel/conv_lstm2d/TensorArrayUnstack/TensorListFromTensor:output_handle:0/model_conv_lstm2d_split_readvariableop_resource1model_conv_lstm2d_split_1_readvariableop_resource1model_conv_lstm2d_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#model_conv_lstm2d_while_body_168584*/
cond'R%
#model_conv_lstm2d_while_cond_168583*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
Bmodel/conv_lstm2d/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          
4model/conv_lstm2d/TensorArrayV2Stack/TensorListStackTensorListStack model/conv_lstm2d/while:output:3Kmodel/conv_lstm2d/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ *
element_dtype0z
'model/conv_lstm2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿs
)model/conv_lstm2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)model/conv_lstm2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
!model/conv_lstm2d/strided_slice_2StridedSlice=model/conv_lstm2d/TensorArrayV2Stack/TensorListStack:tensor:00model/conv_lstm2d/strided_slice_2/stack:output:02model/conv_lstm2d/strided_slice_2/stack_1:output:02model/conv_lstm2d/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_mask
"model/conv_lstm2d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                Ô
model/conv_lstm2d/transpose_1	Transpose=model/conv_lstm2d/TensorArrayV2Stack/TensorListStack:tensor:0+model/conv_lstm2d/transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ }
$model/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          µ
model/time_distributed/ReshapeReshape!model/conv_lstm2d/transpose_1:y:0-model/time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Í
,model/time_distributed/max_pooling2d/MaxPoolMaxPool'model/time_distributed/Reshape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

&model/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿd             Ñ
 model/time_distributed/Reshape_1Reshape5model/time_distributed/max_pooling2d/MaxPool:output:0/model/time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
&model/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¹
 model/time_distributed/Reshape_2Reshape!model/conv_lstm2d/transpose_1:y:0/model/time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
model/conv_lstm2d_1/zeros_like	ZerosLike)model/time_distributed/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd k
)model/conv_lstm2d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :°
model/conv_lstm2d_1/SumSum"model/conv_lstm2d_1/zeros_like:y:02model/conv_lstm2d_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)model/conv_lstm2d_1/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            d
model/conv_lstm2d_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    °
model/conv_lstm2d_1/zerosFill2model/conv_lstm2d_1/zeros/shape_as_tensor:output:0(model/conv_lstm2d_1/zeros/Const:output:0*
T0*&
_output_shapes
:Ñ
model/conv_lstm2d_1/convolutionConv2D model/conv_lstm2d_1/Sum:output:0"model/conv_lstm2d_1/zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

"model/conv_lstm2d_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                À
model/conv_lstm2d_1/transpose	Transpose)model/time_distributed/Reshape_1:output:0+model/conv_lstm2d_1/transpose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ j
model/conv_lstm2d_1/ShapeShape!model/conv_lstm2d_1/transpose:y:0*
T0*
_output_shapes
:q
'model/conv_lstm2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)model/conv_lstm2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)model/conv_lstm2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!model/conv_lstm2d_1/strided_sliceStridedSlice"model/conv_lstm2d_1/Shape:output:00model/conv_lstm2d_1/strided_slice/stack:output:02model/conv_lstm2d_1/strided_slice/stack_1:output:02model/conv_lstm2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/model/conv_lstm2d_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
!model/conv_lstm2d_1/TensorArrayV2TensorListReserve8model/conv_lstm2d_1/TensorArrayV2/element_shape:output:0*model/conv_lstm2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¢
Imodel/conv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          
;model/conv_lstm2d_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!model/conv_lstm2d_1/transpose:y:0Rmodel/conv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)model/conv_lstm2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+model/conv_lstm2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+model/conv_lstm2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Õ
#model/conv_lstm2d_1/strided_slice_1StridedSlice!model/conv_lstm2d_1/transpose:y:02model/conv_lstm2d_1/strided_slice_1/stack:output:04model/conv_lstm2d_1/strided_slice_1/stack_1:output:04model/conv_lstm2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
#model/conv_lstm2d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¢
(model/conv_lstm2d_1/split/ReadVariableOpReadVariableOp1model_conv_lstm2d_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype0ú
model/conv_lstm2d_1/splitSplit,model/conv_lstm2d_1/split/split_dim:output:00model/conv_lstm2d_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitg
%model/conv_lstm2d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¦
*model/conv_lstm2d_1/split_1/ReadVariableOpReadVariableOp3model_conv_lstm2d_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype0
model/conv_lstm2d_1/split_1Split.model/conv_lstm2d_1/split_1/split_dim:output:02model/conv_lstm2d_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitg
%model/conv_lstm2d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
*model/conv_lstm2d_1/split_2/ReadVariableOpReadVariableOp3model_conv_lstm2d_1_split_2_readvariableop_resource*
_output_shapes
:@*
dtype0Ð
model/conv_lstm2d_1/split_2Split.model/conv_lstm2d_1/split_2/split_dim:output:02model/conv_lstm2d_1/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitß
!model/conv_lstm2d_1/convolution_1Conv2D,model/conv_lstm2d_1/strided_slice_1:output:0"model/conv_lstm2d_1/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
²
model/conv_lstm2d_1/BiasAddBiasAdd*model/conv_lstm2d_1/convolution_1:output:0$model/conv_lstm2d_1/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß
!model/conv_lstm2d_1/convolution_2Conv2D,model/conv_lstm2d_1/strided_slice_1:output:0"model/conv_lstm2d_1/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
model/conv_lstm2d_1/BiasAdd_1BiasAdd*model/conv_lstm2d_1/convolution_2:output:0$model/conv_lstm2d_1/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß
!model/conv_lstm2d_1/convolution_3Conv2D,model/conv_lstm2d_1/strided_slice_1:output:0"model/conv_lstm2d_1/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
model/conv_lstm2d_1/BiasAdd_2BiasAdd*model/conv_lstm2d_1/convolution_3:output:0$model/conv_lstm2d_1/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß
!model/conv_lstm2d_1/convolution_4Conv2D,model/conv_lstm2d_1/strided_slice_1:output:0"model/conv_lstm2d_1/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
´
model/conv_lstm2d_1/BiasAdd_3BiasAdd*model/conv_lstm2d_1/convolution_4:output:0$model/conv_lstm2d_1/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ý
!model/conv_lstm2d_1/convolution_5Conv2D(model/conv_lstm2d_1/convolution:output:0$model/conv_lstm2d_1/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ý
!model/conv_lstm2d_1/convolution_6Conv2D(model/conv_lstm2d_1/convolution:output:0$model/conv_lstm2d_1/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ý
!model/conv_lstm2d_1/convolution_7Conv2D(model/conv_lstm2d_1/convolution:output:0$model/conv_lstm2d_1/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ý
!model/conv_lstm2d_1/convolution_8Conv2D(model/conv_lstm2d_1/convolution:output:0$model/conv_lstm2d_1/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
model/conv_lstm2d_1/addAddV2$model/conv_lstm2d_1/BiasAdd:output:0*model/conv_lstm2d_1/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
model/conv_lstm2d_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
model/conv_lstm2d_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
model/conv_lstm2d_1/MulMulmodel/conv_lstm2d_1/add:z:0"model/conv_lstm2d_1/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model/conv_lstm2d_1/Add_1AddV2model/conv_lstm2d_1/Mul:z:0$model/conv_lstm2d_1/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
+model/conv_lstm2d_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
)model/conv_lstm2d_1/clip_by_value/MinimumMinimummodel/conv_lstm2d_1/Add_1:z:04model/conv_lstm2d_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
#model/conv_lstm2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!model/conv_lstm2d_1/clip_by_valueMaximum-model/conv_lstm2d_1/clip_by_value/Minimum:z:0,model/conv_lstm2d_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
model/conv_lstm2d_1/add_2AddV2&model/conv_lstm2d_1/BiasAdd_1:output:0*model/conv_lstm2d_1/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
model/conv_lstm2d_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
model/conv_lstm2d_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
model/conv_lstm2d_1/Mul_1Mulmodel/conv_lstm2d_1/add_2:z:0$model/conv_lstm2d_1/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
model/conv_lstm2d_1/Add_3AddV2model/conv_lstm2d_1/Mul_1:z:0$model/conv_lstm2d_1/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
-model/conv_lstm2d_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
+model/conv_lstm2d_1/clip_by_value_1/MinimumMinimummodel/conv_lstm2d_1/Add_3:z:06model/conv_lstm2d_1/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
%model/conv_lstm2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    É
#model/conv_lstm2d_1/clip_by_value_1Maximum/model/conv_lstm2d_1/clip_by_value_1/Minimum:z:0.model/conv_lstm2d_1/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
model/conv_lstm2d_1/mul_2Mul'model/conv_lstm2d_1/clip_by_value_1:z:0(model/conv_lstm2d_1/convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
model/conv_lstm2d_1/add_4AddV2&model/conv_lstm2d_1/BiasAdd_2:output:0*model/conv_lstm2d_1/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
model/conv_lstm2d_1/ReluRelumodel/conv_lstm2d_1/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
model/conv_lstm2d_1/mul_3Mul%model/conv_lstm2d_1/clip_by_value:z:0&model/conv_lstm2d_1/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model/conv_lstm2d_1/add_5AddV2model/conv_lstm2d_1/mul_2:z:0model/conv_lstm2d_1/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
model/conv_lstm2d_1/add_6AddV2&model/conv_lstm2d_1/BiasAdd_3:output:0*model/conv_lstm2d_1/convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
model/conv_lstm2d_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>`
model/conv_lstm2d_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
model/conv_lstm2d_1/Mul_4Mulmodel/conv_lstm2d_1/add_6:z:0$model/conv_lstm2d_1/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
model/conv_lstm2d_1/Add_7AddV2model/conv_lstm2d_1/Mul_4:z:0$model/conv_lstm2d_1/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
-model/conv_lstm2d_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
+model/conv_lstm2d_1/clip_by_value_2/MinimumMinimummodel/conv_lstm2d_1/Add_7:z:06model/conv_lstm2d_1/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
%model/conv_lstm2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    É
#model/conv_lstm2d_1/clip_by_value_2Maximum/model/conv_lstm2d_1/clip_by_value_2/Minimum:z:0.model/conv_lstm2d_1/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
model/conv_lstm2d_1/Relu_1Relumodel/conv_lstm2d_1/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
model/conv_lstm2d_1/mul_5Mul'model/conv_lstm2d_1/clip_by_value_2:z:0(model/conv_lstm2d_1/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
1model/conv_lstm2d_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ò
#model/conv_lstm2d_1/TensorArrayV2_1TensorListReserve:model/conv_lstm2d_1/TensorArrayV2_1/element_shape:output:0*model/conv_lstm2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
model/conv_lstm2d_1/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,model/conv_lstm2d_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&model/conv_lstm2d_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
model/conv_lstm2d_1/whileWhile/model/conv_lstm2d_1/while/loop_counter:output:05model/conv_lstm2d_1/while/maximum_iterations:output:0!model/conv_lstm2d_1/time:output:0,model/conv_lstm2d_1/TensorArrayV2_1:handle:0(model/conv_lstm2d_1/convolution:output:0(model/conv_lstm2d_1/convolution:output:0*model/conv_lstm2d_1/strided_slice:output:0Kmodel/conv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01model_conv_lstm2d_1_split_readvariableop_resource3model_conv_lstm2d_1_split_1_readvariableop_resource3model_conv_lstm2d_1_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%model_conv_lstm2d_1_while_body_168809*1
cond)R'
%model_conv_lstm2d_1_while_cond_168808*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
Dmodel/conv_lstm2d_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          
6model/conv_lstm2d_1/TensorArrayV2Stack/TensorListStackTensorListStack"model/conv_lstm2d_1/while:output:3Mmodel/conv_lstm2d_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ *
element_dtype0|
)model/conv_lstm2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+model/conv_lstm2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+model/conv_lstm2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ó
#model/conv_lstm2d_1/strided_slice_2StridedSlice?model/conv_lstm2d_1/TensorArrayV2Stack/TensorListStack:tensor:02model/conv_lstm2d_1/strided_slice_2/stack:output:04model/conv_lstm2d_1/strided_slice_2/stack_1:output:04model/conv_lstm2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
$model/conv_lstm2d_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                Ú
model/conv_lstm2d_1/transpose_1	Transpose?model/conv_lstm2d_1/TensorArrayV2Stack/TensorListStack:tensor:0-model/conv_lstm2d_1/transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
&model/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          »
 model/time_distributed_1/ReshapeReshape#model/conv_lstm2d_1/transpose_1:y:0/model/time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
,model/time_distributed_1/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
.model/time_distributed_1/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Æ
*model/time_distributed_1/up_sampling2d/mulMul5model/time_distributed_1/up_sampling2d/Const:output:07model/time_distributed_1/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:
Cmodel/time_distributed_1/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor)model/time_distributed_1/Reshape:output:0.model/time_distributed_1/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
half_pixel_centers(
(model/time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿd   @          ô
"model/time_distributed_1/Reshape_1ReshapeTmodel/time_distributed_1/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:01model/time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
(model/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¿
"model/time_distributed_1/Reshape_2Reshape#model/conv_lstm2d_1/transpose_1:y:01model/time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
model/conv2d/Conv2D/ShapeShape+model/time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:q
'model/conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
)model/conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿs
)model/conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
!model/conv2d/Conv2D/strided_sliceStridedSlice"model/conv2d/Conv2D/Shape:output:00model/conv2d/Conv2D/strided_slice/stack:output:02model/conv2d/Conv2D/strided_slice/stack_1:output:02model/conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskz
!model/conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¹
model/conv2d/Conv2D/ReshapeReshape+model/time_distributed_1/Reshape_1:output:0*model/conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¤
)model/conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp2model_conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ß
model/conv2d/Conv2D/Conv2DConv2D$model/conv2d/Conv2D/Reshape:output:01model/conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
#model/conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          j
model/conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿØ
model/conv2d/Conv2D/concatConcatV2*model/conv2d/Conv2D/strided_slice:output:0,model/conv2d/Conv2D/concat/values_1:output:0(model/conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:°
model/conv2d/Conv2D/Reshape_1Reshape#model/conv2d/Conv2D/Conv2D:output:0#model/conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ {
%model/conv2d/squeeze_batch_dims/ShapeShape&model/conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:}
3model/conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5model/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
5model/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-model/conv2d/squeeze_batch_dims/strided_sliceStridedSlice.model/conv2d/squeeze_batch_dims/Shape:output:0<model/conv2d/squeeze_batch_dims/strided_slice/stack:output:0>model/conv2d/squeeze_batch_dims/strided_slice/stack_1:output:0>model/conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
-model/conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ì
'model/conv2d/squeeze_batch_dims/ReshapeReshape&model/conv2d/Conv2D/Reshape_1:output:06model/conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ²
6model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp?model_conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Þ
'model/conv2d/squeeze_batch_dims/BiasAddBiasAdd0model/conv2d/squeeze_batch_dims/Reshape:output:0>model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
/model/conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          v
+model/conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
&model/conv2d/squeeze_batch_dims/concatConcatV26model/conv2d/squeeze_batch_dims/strided_slice:output:08model/conv2d/squeeze_batch_dims/concat/values_1:output:04model/conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Õ
)model/conv2d/squeeze_batch_dims/Reshape_1Reshape0model/conv2d/squeeze_batch_dims/BiasAdd:output:0/model/conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
model/conv2d/ReluRelu2model/conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ j
model/conv2d_1/Conv2D/ShapeShapemodel/conv2d/Relu:activations:0*
T0*
_output_shapes
:s
)model/conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+model/conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿu
+model/conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
#model/conv2d_1/Conv2D/strided_sliceStridedSlice$model/conv2d_1/Conv2D/Shape:output:02model/conv2d_1/Conv2D/strided_slice/stack:output:04model/conv2d_1/Conv2D/strided_slice/stack_1:output:04model/conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
#model/conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ±
model/conv2d_1/Conv2D/ReshapeReshapemodel/conv2d/Relu:activations:0,model/conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¨
+model/conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp4model_conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0å
model/conv2d_1/Conv2D/Conv2DConv2D&model/conv2d_1/Conv2D/Reshape:output:03model/conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*
paddingSAME*
strides
z
%model/conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   l
!model/conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿà
model/conv2d_1/Conv2D/concatConcatV2,model/conv2d_1/Conv2D/strided_slice:output:0.model/conv2d_1/Conv2D/concat/values_1:output:0*model/conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¶
model/conv2d_1/Conv2D/Reshape_1Reshape%model/conv2d_1/Conv2D/Conv2D:output:0%model/conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 

'model/conv2d_1/squeeze_batch_dims/ShapeShape(model/conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:
5model/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7model/conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
7model/conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/model/conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice0model/conv2d_1/squeeze_batch_dims/Shape:output:0>model/conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0@model/conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0@model/conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
/model/conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@       
   Ò
)model/conv2d_1/squeeze_batch_dims/ReshapeReshape(model/conv2d_1/Conv2D/Reshape_1:output:08model/conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
¶
8model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpAmodel_conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ä
)model/conv2d_1/squeeze_batch_dims/BiasAddBiasAdd2model/conv2d_1/squeeze_batch_dims/Reshape:output:0@model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 

1model/conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   x
-model/conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
(model/conv2d_1/squeeze_batch_dims/concatConcatV28model/conv2d_1/squeeze_batch_dims/strided_slice:output:0:model/conv2d_1/squeeze_batch_dims/concat/values_1:output:06model/conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Û
+model/conv2d_1/squeeze_batch_dims/Reshape_1Reshape2model/conv2d_1/squeeze_batch_dims/BiasAdd:output:01model/conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 

model/conv2d_1/SoftmaxSoftmax4model/conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
{
IdentityIdentity model/conv2d_1/Softmax:softmax:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
Î
NoOpNoOp*^model/conv2d/Conv2D/Conv2D/ReadVariableOp7^model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp,^model/conv2d_1/Conv2D/Conv2D/ReadVariableOp9^model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp'^model/conv_lstm2d/split/ReadVariableOp)^model/conv_lstm2d/split_1/ReadVariableOp)^model/conv_lstm2d/split_2/ReadVariableOp^model/conv_lstm2d/while)^model/conv_lstm2d_1/split/ReadVariableOp+^model/conv_lstm2d_1/split_1/ReadVariableOp+^model/conv_lstm2d_1/split_2/ReadVariableOp^model/conv_lstm2d_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 2V
)model/conv2d/Conv2D/Conv2D/ReadVariableOp)model/conv2d/Conv2D/Conv2D/ReadVariableOp2p
6model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp6model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+model/conv2d_1/Conv2D/Conv2D/ReadVariableOp+model/conv2d_1/Conv2D/Conv2D/ReadVariableOp2t
8model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp8model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2P
&model/conv_lstm2d/split/ReadVariableOp&model/conv_lstm2d/split/ReadVariableOp2T
(model/conv_lstm2d/split_1/ReadVariableOp(model/conv_lstm2d/split_1/ReadVariableOp2T
(model/conv_lstm2d/split_2/ReadVariableOp(model/conv_lstm2d/split_2/ReadVariableOp22
model/conv_lstm2d/whilemodel/conv_lstm2d/while2T
(model/conv_lstm2d_1/split/ReadVariableOp(model/conv_lstm2d_1/split/ReadVariableOp2X
*model/conv_lstm2d_1/split_1/ReadVariableOp*model/conv_lstm2d_1/split_1/ReadVariableOp2X
*model/conv_lstm2d_1/split_2/ReadVariableOp*model/conv_lstm2d_1/split_2/ReadVariableOp26
model/conv_lstm2d_1/whilemodel/conv_lstm2d_1/while:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
!
_user_specified_name	input_1
Z
ë
while_body_173887
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0:@A
'while_split_1_readvariableop_resource_0:@5
'while_split_2_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource:@?
%while_split_1_readvariableop_resource:@3
%while_split_2_readvariableop_resource:@¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ö
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_169478

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ×
max_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_169430\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B : S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&max_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs
Æd
Þ
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_173791
inputs_07
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢whileh

zeros_like	ZerosLikeinputs_0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    t
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_173665*
condR
while_cond_173664*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          Ó
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                §
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
IdentityIdentitytranspose_1:y:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Æd
Þ
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_173569
inputs_07
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢whileh

zeros_like	ZerosLikeinputs_0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    t
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_173443*
condR
while_cond_173442*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          Ó
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                §
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
IdentityIdentitytranspose_1:y:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Z
ë
while_body_173131
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0: A
'while_split_1_readvariableop_resource_0: 5
'while_split_2_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource: ?
%while_split_1_readvariableop_resource: 3
%while_split_2_readvariableop_resource: ¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
ö

 
&__inference_model_layer_call_fn_171156
input_1!
unknown: #
	unknown_0: 
	unknown_1: #
	unknown_2:@#
	unknown_3:@
	unknown_4:@#
	unknown_5:
	unknown_6:#
	unknown_7:

	unknown_8:

identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_171108{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
!
_user_specified_name	input_1
·
Æ
,__inference_conv_lstm2d_layer_call_fn_172366

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_170196{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd@ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
»
È
.__inference_conv_lstm2d_1_layer_call_fn_173347

inputs!
unknown:@#
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_170796{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
 
_user_specified_nameinputs
Å=

L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_169585

inputs

states
states_17
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity

identity_1

identity_2¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
Z
ë
while_body_174109
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0:@A
'while_split_1_readvariableop_resource_0:@5
'while_split_2_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource:@?
%while_split_1_readvariableop_resource:@3
%while_split_2_readvariableop_resource:@¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Æ{
«
#model_conv_lstm2d_while_body_168584@
<model_conv_lstm2d_while_model_conv_lstm2d_while_loop_counterF
Bmodel_conv_lstm2d_while_model_conv_lstm2d_while_maximum_iterations'
#model_conv_lstm2d_while_placeholder)
%model_conv_lstm2d_while_placeholder_1)
%model_conv_lstm2d_while_placeholder_2)
%model_conv_lstm2d_while_placeholder_3=
9model_conv_lstm2d_while_model_conv_lstm2d_strided_slice_0{
wmodel_conv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0Q
7model_conv_lstm2d_while_split_readvariableop_resource_0: S
9model_conv_lstm2d_while_split_1_readvariableop_resource_0: G
9model_conv_lstm2d_while_split_2_readvariableop_resource_0: $
 model_conv_lstm2d_while_identity&
"model_conv_lstm2d_while_identity_1&
"model_conv_lstm2d_while_identity_2&
"model_conv_lstm2d_while_identity_3&
"model_conv_lstm2d_while_identity_4&
"model_conv_lstm2d_while_identity_5;
7model_conv_lstm2d_while_model_conv_lstm2d_strided_slicey
umodel_conv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_tensorarrayunstack_tensorlistfromtensorO
5model_conv_lstm2d_while_split_readvariableop_resource: Q
7model_conv_lstm2d_while_split_1_readvariableop_resource: E
7model_conv_lstm2d_while_split_2_readvariableop_resource: ¢,model/conv_lstm2d/while/split/ReadVariableOp¢.model/conv_lstm2d/while/split_1/ReadVariableOp¢.model/conv_lstm2d/while/split_2/ReadVariableOp¢
Imodel/conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          
;model/conv_lstm2d/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwmodel_conv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0#model_conv_lstm2d_while_placeholderRmodel/conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0i
'model/conv_lstm2d/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¬
,model/conv_lstm2d/while/split/ReadVariableOpReadVariableOp7model_conv_lstm2d_while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0
model/conv_lstm2d/while/splitSplit0model/conv_lstm2d/while/split/split_dim:output:04model/conv_lstm2d/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitk
)model/conv_lstm2d/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :°
.model/conv_lstm2d/while/split_1/ReadVariableOpReadVariableOp9model_conv_lstm2d_while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0
model/conv_lstm2d/while/split_1Split2model/conv_lstm2d/while/split_1/split_dim:output:06model/conv_lstm2d/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitk
)model/conv_lstm2d/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ¤
.model/conv_lstm2d/while/split_2/ReadVariableOpReadVariableOp9model_conv_lstm2d_while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0Ü
model/conv_lstm2d/while/split_2Split2model/conv_lstm2d/while/split_2/split_dim:output:06model/conv_lstm2d/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitû
#model/conv_lstm2d/while/convolutionConv2DBmodel/conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0&model/conv_lstm2d/while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¼
model/conv_lstm2d/while/BiasAddBiasAdd,model/conv_lstm2d/while/convolution:output:0(model/conv_lstm2d/while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ý
%model/conv_lstm2d/while/convolution_1Conv2DBmodel/conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0&model/conv_lstm2d/while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
À
!model/conv_lstm2d/while/BiasAdd_1BiasAdd.model/conv_lstm2d/while/convolution_1:output:0(model/conv_lstm2d/while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ý
%model/conv_lstm2d/while/convolution_2Conv2DBmodel/conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0&model/conv_lstm2d/while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
À
!model/conv_lstm2d/while/BiasAdd_2BiasAdd.model/conv_lstm2d/while/convolution_2:output:0(model/conv_lstm2d/while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ý
%model/conv_lstm2d/while/convolution_3Conv2DBmodel/conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0&model/conv_lstm2d/while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
À
!model/conv_lstm2d/while/BiasAdd_3BiasAdd.model/conv_lstm2d/while/convolution_3:output:0(model/conv_lstm2d/while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ â
%model/conv_lstm2d/while/convolution_4Conv2D%model_conv_lstm2d_while_placeholder_2(model/conv_lstm2d/while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
â
%model/conv_lstm2d/while/convolution_5Conv2D%model_conv_lstm2d_while_placeholder_2(model/conv_lstm2d/while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
â
%model/conv_lstm2d/while/convolution_6Conv2D%model_conv_lstm2d_while_placeholder_2(model/conv_lstm2d/while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
â
%model/conv_lstm2d/while/convolution_7Conv2D%model_conv_lstm2d_while_placeholder_2(model/conv_lstm2d/while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¸
model/conv_lstm2d/while/addAddV2(model/conv_lstm2d/while/BiasAdd:output:0.model/conv_lstm2d/while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
model/conv_lstm2d/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>d
model/conv_lstm2d/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?¥
model/conv_lstm2d/while/MulMulmodel/conv_lstm2d/while/add:z:0&model/conv_lstm2d/while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ «
model/conv_lstm2d/while/Add_1AddV2model/conv_lstm2d/while/Mul:z:0(model/conv_lstm2d/while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
/model/conv_lstm2d/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ï
-model/conv_lstm2d/while/clip_by_value/MinimumMinimum!model/conv_lstm2d/while/Add_1:z:08model/conv_lstm2d/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ l
'model/conv_lstm2d/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ï
%model/conv_lstm2d/while/clip_by_valueMaximum1model/conv_lstm2d/while/clip_by_value/Minimum:z:00model/conv_lstm2d/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¼
model/conv_lstm2d/while/add_2AddV2*model/conv_lstm2d/while/BiasAdd_1:output:0.model/conv_lstm2d/while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
model/conv_lstm2d/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>d
model/conv_lstm2d/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?«
model/conv_lstm2d/while/Mul_1Mul!model/conv_lstm2d/while/add_2:z:0(model/conv_lstm2d/while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ­
model/conv_lstm2d/while/Add_3AddV2!model/conv_lstm2d/while/Mul_1:z:0(model/conv_lstm2d/while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
1model/conv_lstm2d/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ó
/model/conv_lstm2d/while/clip_by_value_1/MinimumMinimum!model/conv_lstm2d/while/Add_3:z:0:model/conv_lstm2d/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ n
)model/conv_lstm2d/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Õ
'model/conv_lstm2d/while/clip_by_value_1Maximum3model/conv_lstm2d/while/clip_by_value_1/Minimum:z:02model/conv_lstm2d/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ²
model/conv_lstm2d/while/mul_2Mul+model/conv_lstm2d/while/clip_by_value_1:z:0%model_conv_lstm2d_while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¼
model/conv_lstm2d/while/add_4AddV2*model/conv_lstm2d/while/BiasAdd_2:output:0.model/conv_lstm2d/while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
model/conv_lstm2d/while/ReluRelu!model/conv_lstm2d/while/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ µ
model/conv_lstm2d/while/mul_3Mul)model/conv_lstm2d/while/clip_by_value:z:0*model/conv_lstm2d/while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¦
model/conv_lstm2d/while/add_5AddV2!model/conv_lstm2d/while/mul_2:z:0!model/conv_lstm2d/while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¼
model/conv_lstm2d/while/add_6AddV2*model/conv_lstm2d/while/BiasAdd_3:output:0.model/conv_lstm2d/while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
model/conv_lstm2d/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>d
model/conv_lstm2d/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?«
model/conv_lstm2d/while/Mul_4Mul!model/conv_lstm2d/while/add_6:z:0(model/conv_lstm2d/while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ­
model/conv_lstm2d/while/Add_7AddV2!model/conv_lstm2d/while/Mul_4:z:0(model/conv_lstm2d/while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
1model/conv_lstm2d/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ó
/model/conv_lstm2d/while/clip_by_value_2/MinimumMinimum!model/conv_lstm2d/while/Add_7:z:0:model/conv_lstm2d/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ n
)model/conv_lstm2d/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Õ
'model/conv_lstm2d/while/clip_by_value_2Maximum3model/conv_lstm2d/while/clip_by_value_2/Minimum:z:02model/conv_lstm2d/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
model/conv_lstm2d/while/Relu_1Relu!model/conv_lstm2d/while/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¹
model/conv_lstm2d/while/mul_5Mul+model/conv_lstm2d/while/clip_by_value_2:z:0,model/conv_lstm2d/while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
<model/conv_lstm2d/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%model_conv_lstm2d_while_placeholder_1#model_conv_lstm2d_while_placeholder!model/conv_lstm2d/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒa
model/conv_lstm2d/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :
model/conv_lstm2d/while/add_8AddV2#model_conv_lstm2d_while_placeholder(model/conv_lstm2d/while/add_8/y:output:0*
T0*
_output_shapes
: a
model/conv_lstm2d/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :¯
model/conv_lstm2d/while/add_9AddV2<model_conv_lstm2d_while_model_conv_lstm2d_while_loop_counter(model/conv_lstm2d/while/add_9/y:output:0*
T0*
_output_shapes
: 
 model/conv_lstm2d/while/IdentityIdentity!model/conv_lstm2d/while/add_9:z:0^model/conv_lstm2d/while/NoOp*
T0*
_output_shapes
: ²
"model/conv_lstm2d/while/Identity_1IdentityBmodel_conv_lstm2d_while_model_conv_lstm2d_while_maximum_iterations^model/conv_lstm2d/while/NoOp*
T0*
_output_shapes
: 
"model/conv_lstm2d/while/Identity_2Identity!model/conv_lstm2d/while/add_8:z:0^model/conv_lstm2d/while/NoOp*
T0*
_output_shapes
: Ï
"model/conv_lstm2d/while/Identity_3IdentityLmodel/conv_lstm2d/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/conv_lstm2d/while/NoOp*
T0*
_output_shapes
: :éèÒª
"model/conv_lstm2d/while/Identity_4Identity!model/conv_lstm2d/while/mul_5:z:0^model/conv_lstm2d/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
"model/conv_lstm2d/while/Identity_5Identity!model/conv_lstm2d/while/add_5:z:0^model/conv_lstm2d/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ï
model/conv_lstm2d/while/NoOpNoOp-^model/conv_lstm2d/while/split/ReadVariableOp/^model/conv_lstm2d/while/split_1/ReadVariableOp/^model/conv_lstm2d/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "M
 model_conv_lstm2d_while_identity)model/conv_lstm2d/while/Identity:output:0"Q
"model_conv_lstm2d_while_identity_1+model/conv_lstm2d/while/Identity_1:output:0"Q
"model_conv_lstm2d_while_identity_2+model/conv_lstm2d/while/Identity_2:output:0"Q
"model_conv_lstm2d_while_identity_3+model/conv_lstm2d/while/Identity_3:output:0"Q
"model_conv_lstm2d_while_identity_4+model/conv_lstm2d/while/Identity_4:output:0"Q
"model_conv_lstm2d_while_identity_5+model/conv_lstm2d/while/Identity_5:output:0"t
7model_conv_lstm2d_while_model_conv_lstm2d_strided_slice9model_conv_lstm2d_while_model_conv_lstm2d_strided_slice_0"t
7model_conv_lstm2d_while_split_1_readvariableop_resource9model_conv_lstm2d_while_split_1_readvariableop_resource_0"t
7model_conv_lstm2d_while_split_2_readvariableop_resource9model_conv_lstm2d_while_split_2_readvariableop_resource_0"p
5model_conv_lstm2d_while_split_readvariableop_resource7model_conv_lstm2d_while_split_readvariableop_resource_0"ð
umodel_conv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_tensorarrayunstack_tensorlistfromtensorwmodel_conv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_model_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 2\
,model/conv_lstm2d/while/split/ReadVariableOp,model/conv_lstm2d/while/split/ReadVariableOp2`
.model/conv_lstm2d/while/split_1/ReadVariableOp.model/conv_lstm2d/while/split_1/ReadVariableOp2`
.model/conv_lstm2d/while/split_2/ReadVariableOp.model/conv_lstm2d/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
Z
ë
while_body_173665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0:@A
'while_split_1_readvariableop_resource_0:@5
'while_split_2_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource:@?
%while_split_1_readvariableop_resource:@3
%while_split_2_readvariableop_resource:@¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ú
Ù
conv_lstm2d_1_while_cond_1721118
4conv_lstm2d_1_while_conv_lstm2d_1_while_loop_counter>
:conv_lstm2d_1_while_conv_lstm2d_1_while_maximum_iterations#
conv_lstm2d_1_while_placeholder%
!conv_lstm2d_1_while_placeholder_1%
!conv_lstm2d_1_while_placeholder_2%
!conv_lstm2d_1_while_placeholder_38
4conv_lstm2d_1_while_less_conv_lstm2d_1_strided_sliceP
Lconv_lstm2d_1_while_conv_lstm2d_1_while_cond_172111___redundant_placeholder0P
Lconv_lstm2d_1_while_conv_lstm2d_1_while_cond_172111___redundant_placeholder1P
Lconv_lstm2d_1_while_conv_lstm2d_1_while_cond_172111___redundant_placeholder2P
Lconv_lstm2d_1_while_conv_lstm2d_1_while_cond_172111___redundant_placeholder3 
conv_lstm2d_1_while_identity

conv_lstm2d_1/while/LessLessconv_lstm2d_1_while_placeholder4conv_lstm2d_1_while_less_conv_lstm2d_1_strided_slice*
T0*
_output_shapes
: g
conv_lstm2d_1/while/IdentityIdentityconv_lstm2d_1/while/Less:z:0*
T0
*
_output_shapes
: "E
conv_lstm2d_1_while_identity%conv_lstm2d_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
£$

A__inference_model_layer_call_and_return_conditional_losses_171224
input_1,
conv_lstm2d_171193: ,
conv_lstm2d_171195:  
conv_lstm2d_171197: .
conv_lstm2d_1_171203:@.
conv_lstm2d_1_171205:@"
conv_lstm2d_1_171207:@'
conv2d_171213:
conv2d_171215:)
conv2d_1_171218:

conv2d_1_171220:

identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢#conv_lstm2d/StatefulPartitionedCall¢%conv_lstm2d_1/StatefulPartitionedCall¢
#conv_lstm2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_lstm2d_171193conv_lstm2d_171195conv_lstm2d_171197*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_171038ý
 time_distributed/PartitionedCallPartitionedCall,conv_lstm2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_169478w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ´
time_distributed/ReshapeReshape,conv_lstm2d/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Î
%conv_lstm2d_1/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0conv_lstm2d_1_171203conv_lstm2d_1_171205conv_lstm2d_1_171207*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_170796
"time_distributed_1/PartitionedCallPartitionedCall.conv_lstm2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_169967y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          º
time_distributed_1/ReshapeReshape.conv_lstm2d_1/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:0conv2d_171213conv2d_171215*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_170472 
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_171218conv2d_1_171220*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_170511
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
Ø
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall$^conv_lstm2d/StatefulPartitionedCall&^conv_lstm2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2J
#conv_lstm2d/StatefulPartitionedCall#conv_lstm2d/StatefulPartitionedCall2N
%conv_lstm2d_1/StatefulPartitionedCall%conv_lstm2d_1/StatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
!
_user_specified_name	input_1
°b
Ú
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_173037

inputs7
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢while]

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                u
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_172911*
condR
while_cond_172910*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ê
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ j
IdentityIdentitytranspose_1:y:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
Ñ
Á
while_cond_173130
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_173130___redundant_placeholder04
0while_while_cond_173130___redundant_placeholder14
0while_while_cond_173130___redundant_placeholder24
0while_while_cond_173130___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
ã

1__inference_conv_lstm_cell_1_layer_call_fn_174582

inputs
states_0
states_1!
unknown:@#
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_169585w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
¶
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_173303

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B : S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_169430

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
Ê
.__inference_conv_lstm2d_1_layer_call_fn_173325
inputs_0!
unknown:@#
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_169894
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Ñ
Á
while_cond_169118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_169118___redundant_placeholder04
0while_while_cond_169118___redundant_placeholder14
0while_while_cond_169118___redundant_placeholder24
0while_while_cond_169118___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
Ñ
Á
while_cond_172690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_172690___redundant_placeholder04
0while_while_cond_172690___redundant_placeholder14
0while_while_cond_172690___redundant_placeholder24
0while_while_cond_172690___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
Ñ=

J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_174480

inputs
states_0
states_17
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity

identity_1

identity_2¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ `
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
states/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
states/1
·
Æ
,__inference_conv_lstm2d_layer_call_fn_172377

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_171038{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd@ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
¢
±
conv_lstm2d_while_cond_1718864
0conv_lstm2d_while_conv_lstm2d_while_loop_counter:
6conv_lstm2d_while_conv_lstm2d_while_maximum_iterations!
conv_lstm2d_while_placeholder#
conv_lstm2d_while_placeholder_1#
conv_lstm2d_while_placeholder_2#
conv_lstm2d_while_placeholder_34
0conv_lstm2d_while_less_conv_lstm2d_strided_sliceL
Hconv_lstm2d_while_conv_lstm2d_while_cond_171886___redundant_placeholder0L
Hconv_lstm2d_while_conv_lstm2d_while_cond_171886___redundant_placeholder1L
Hconv_lstm2d_while_conv_lstm2d_while_cond_171886___redundant_placeholder2L
Hconv_lstm2d_while_conv_lstm2d_while_cond_171886___redundant_placeholder3
conv_lstm2d_while_identity

conv_lstm2d/while/LessLessconv_lstm2d_while_placeholder0conv_lstm2d_while_less_conv_lstm2d_strided_slice*
T0*
_output_shapes
: c
conv_lstm2d/while/IdentityIdentityconv_lstm2d/while/Less:z:0*
T0
*
_output_shapes
: "A
conv_lstm2d_while_identity#conv_lstm2d/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
·
J
.__inference_max_pooling2d_layer_call_fn_174560

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_169430
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½!

while_body_169344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_169368_0: (
while_169370_0: 
while_169372_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_169368: &
while_169370: 
while_169372: ¢while/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_169368_0while_169370_0while_169372_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_169293Ï
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ l

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
while_169368while_169368_0"
while_169370while_169370_0"
while_169372while_169372_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
ó


&__inference_model_layer_call_fn_171280

inputs!
unknown: #
	unknown_0: 
	unknown_1: #
	unknown_2:@#
	unknown_3:@
	unknown_4:@#
	unknown_5:
	unknown_6:#
	unknown_7:

	unknown_8:

identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_171108{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
ó


&__inference_model_layer_call_fn_171255

inputs!
unknown: #
	unknown_0: 
	unknown_1: #
	unknown_2:@#
	unknown_3:@
	unknown_4:@#
	unknown_5:
	unknown_6:#
	unknown_7:

	unknown_8:

identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_170518{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
Ô


$__inference_signature_wrapper_172333
input_1!
unknown: #
	unknown_0: 
	unknown_1: #
	unknown_2:@#
	unknown_3:@
	unknown_4:@#
	unknown_5:
	unknown_6:#
	unknown_7:

	unknown_8:

identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_169003{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
!
_user_specified_name	input_1
Z
ë
while_body_170670
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0:@A
'while_split_1_readvariableop_resource_0:@5
'while_split_2_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource:@?
%while_split_1_readvariableop_resource:@3
%while_split_2_readvariableop_resource:@¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:@*
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ã=

J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_169105

inputs

states
states_17
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity

identity_1

identity_2¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ `
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_namestates:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_namestates

Ñ
%model_conv_lstm2d_1_while_cond_168808D
@model_conv_lstm2d_1_while_model_conv_lstm2d_1_while_loop_counterJ
Fmodel_conv_lstm2d_1_while_model_conv_lstm2d_1_while_maximum_iterations)
%model_conv_lstm2d_1_while_placeholder+
'model_conv_lstm2d_1_while_placeholder_1+
'model_conv_lstm2d_1_while_placeholder_2+
'model_conv_lstm2d_1_while_placeholder_3D
@model_conv_lstm2d_1_while_less_model_conv_lstm2d_1_strided_slice\
Xmodel_conv_lstm2d_1_while_model_conv_lstm2d_1_while_cond_168808___redundant_placeholder0\
Xmodel_conv_lstm2d_1_while_model_conv_lstm2d_1_while_cond_168808___redundant_placeholder1\
Xmodel_conv_lstm2d_1_while_model_conv_lstm2d_1_while_cond_168808___redundant_placeholder2\
Xmodel_conv_lstm2d_1_while_model_conv_lstm2d_1_while_cond_168808___redundant_placeholder3&
"model_conv_lstm2d_1_while_identity
°
model/conv_lstm2d_1/while/LessLess%model_conv_lstm2d_1_while_placeholder@model_conv_lstm2d_1_while_less_model_conv_lstm2d_1_strided_slice*
T0*
_output_shapes
: s
"model/conv_lstm2d_1/while/IdentityIdentity"model/conv_lstm2d_1/while/Less:z:0*
T0
*
_output_shapes
: "Q
"model_conv_lstm2d_1_while_identity+model/conv_lstm2d_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¡%
¯
B__inference_conv2d_layer_call_and_return_conditional_losses_170472

inputs?
%conv2d_conv2d_readvariableop_resource:@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢Conv2D/Conv2D/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¸
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¥
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0·
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:®
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd@ : : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs

O
3__inference_time_distributed_1_layer_call_fn_174245

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_169967u
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ã

1__inference_conv_lstm_cell_1_layer_call_fn_174599

inputs
states_0
states_1!
unknown:@#
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_169773w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1

j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_169967

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ é
up_sampling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_169919\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&up_sampling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ú
Ù
conv_lstm2d_1_while_cond_1715988
4conv_lstm2d_1_while_conv_lstm2d_1_while_loop_counter>
:conv_lstm2d_1_while_conv_lstm2d_1_while_maximum_iterations#
conv_lstm2d_1_while_placeholder%
!conv_lstm2d_1_while_placeholder_1%
!conv_lstm2d_1_while_placeholder_2%
!conv_lstm2d_1_while_placeholder_38
4conv_lstm2d_1_while_less_conv_lstm2d_1_strided_sliceP
Lconv_lstm2d_1_while_conv_lstm2d_1_while_cond_171598___redundant_placeholder0P
Lconv_lstm2d_1_while_conv_lstm2d_1_while_cond_171598___redundant_placeholder1P
Lconv_lstm2d_1_while_conv_lstm2d_1_while_cond_171598___redundant_placeholder2P
Lconv_lstm2d_1_while_conv_lstm2d_1_while_cond_171598___redundant_placeholder3 
conv_lstm2d_1_while_identity

conv_lstm2d_1/while/LessLessconv_lstm2d_1_while_placeholder4conv_lstm2d_1_while_less_conv_lstm2d_1_strided_slice*
T0*
_output_shapes
: g
conv_lstm2d_1/while/IdentityIdentityconv_lstm2d_1/while/Less:z:0*
T0
*
_output_shapes
: "E
conv_lstm2d_1_while_identity%conv_lstm2d_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
þc
Ü
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_174235

inputs7
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢while]

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    t
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                u
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_174109*
condR
while_cond_174108*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          Ê
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd j
IdentityIdentitytranspose_1:y:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
 
_user_specified_nameinputs
æ
Ê
.__inference_conv_lstm2d_1_layer_call_fn_173314
inputs_0!
unknown:@#
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_169667
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Å=

L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_169773

inputs

states
states_17
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity

identity_1

identity_2¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
Ã=

J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_169293

inputs

states
states_17
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity

identity_1

identity_2¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ `
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_namestates:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_namestates
Ñ
Á
while_cond_170301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_170301___redundant_placeholder04
0while_while_cond_170301___redundant_placeholder14
0while_while_cond_170301___redundant_placeholder24
0while_while_cond_170301___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ºp
ë

conv_lstm2d_while_body_1713744
0conv_lstm2d_while_conv_lstm2d_while_loop_counter:
6conv_lstm2d_while_conv_lstm2d_while_maximum_iterations!
conv_lstm2d_while_placeholder#
conv_lstm2d_while_placeholder_1#
conv_lstm2d_while_placeholder_2#
conv_lstm2d_while_placeholder_31
-conv_lstm2d_while_conv_lstm2d_strided_slice_0o
kconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0K
1conv_lstm2d_while_split_readvariableop_resource_0: M
3conv_lstm2d_while_split_1_readvariableop_resource_0: A
3conv_lstm2d_while_split_2_readvariableop_resource_0: 
conv_lstm2d_while_identity 
conv_lstm2d_while_identity_1 
conv_lstm2d_while_identity_2 
conv_lstm2d_while_identity_3 
conv_lstm2d_while_identity_4 
conv_lstm2d_while_identity_5/
+conv_lstm2d_while_conv_lstm2d_strided_slicem
iconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensorI
/conv_lstm2d_while_split_readvariableop_resource: K
1conv_lstm2d_while_split_1_readvariableop_resource: ?
1conv_lstm2d_while_split_2_readvariableop_resource: ¢&conv_lstm2d/while/split/ReadVariableOp¢(conv_lstm2d/while/split_1/ReadVariableOp¢(conv_lstm2d/while/split_2/ReadVariableOp
Cconv_lstm2d/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ê
5conv_lstm2d/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0conv_lstm2d_while_placeholderLconv_lstm2d/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0c
!conv_lstm2d/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&conv_lstm2d/while/split/ReadVariableOpReadVariableOp1conv_lstm2d_while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0ô
conv_lstm2d/while/splitSplit*conv_lstm2d/while/split/split_dim:output:0.conv_lstm2d/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splite
#conv_lstm2d/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¤
(conv_lstm2d/while/split_1/ReadVariableOpReadVariableOp3conv_lstm2d_while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0ú
conv_lstm2d/while/split_1Split,conv_lstm2d/while/split_1/split_dim:output:00conv_lstm2d/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splite
#conv_lstm2d/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(conv_lstm2d/while/split_2/ReadVariableOpReadVariableOp3conv_lstm2d_while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0Ê
conv_lstm2d/while/split_2Split,conv_lstm2d/while/split_2/split_dim:output:00conv_lstm2d/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splité
conv_lstm2d/while/convolutionConv2D<conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0 conv_lstm2d/while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
ª
conv_lstm2d/while/BiasAddBiasAdd&conv_lstm2d/while/convolution:output:0"conv_lstm2d/while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ë
conv_lstm2d/while/convolution_1Conv2D<conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0 conv_lstm2d/while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
conv_lstm2d/while/BiasAdd_1BiasAdd(conv_lstm2d/while/convolution_1:output:0"conv_lstm2d/while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ë
conv_lstm2d/while/convolution_2Conv2D<conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0 conv_lstm2d/while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
conv_lstm2d/while/BiasAdd_2BiasAdd(conv_lstm2d/while/convolution_2:output:0"conv_lstm2d/while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ë
conv_lstm2d/while/convolution_3Conv2D<conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0 conv_lstm2d/while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
conv_lstm2d/while/BiasAdd_3BiasAdd(conv_lstm2d/while/convolution_3:output:0"conv_lstm2d/while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ð
conv_lstm2d/while/convolution_4Conv2Dconv_lstm2d_while_placeholder_2"conv_lstm2d/while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Ð
conv_lstm2d/while/convolution_5Conv2Dconv_lstm2d_while_placeholder_2"conv_lstm2d/while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Ð
conv_lstm2d/while/convolution_6Conv2Dconv_lstm2d_while_placeholder_2"conv_lstm2d/while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Ð
conv_lstm2d/while/convolution_7Conv2Dconv_lstm2d_while_placeholder_2"conv_lstm2d/while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¦
conv_lstm2d/while/addAddV2"conv_lstm2d/while/BiasAdd:output:0(conv_lstm2d/while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
conv_lstm2d/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
conv_lstm2d/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/while/MulMulconv_lstm2d/while/add:z:0 conv_lstm2d/while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/Add_1AddV2conv_lstm2d/while/Mul:z:0"conv_lstm2d/while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ n
)conv_lstm2d/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?½
'conv_lstm2d/while/clip_by_value/MinimumMinimumconv_lstm2d/while/Add_1:z:02conv_lstm2d/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ f
!conv_lstm2d/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
conv_lstm2d/while/clip_by_valueMaximum+conv_lstm2d/while/clip_by_value/Minimum:z:0*conv_lstm2d/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
conv_lstm2d/while/add_2AddV2$conv_lstm2d/while/BiasAdd_1:output:0(conv_lstm2d/while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
conv_lstm2d/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
conv_lstm2d/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/while/Mul_1Mulconv_lstm2d/while/add_2:z:0"conv_lstm2d/while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/Add_3AddV2conv_lstm2d/while/Mul_1:z:0"conv_lstm2d/while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
+conv_lstm2d/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
)conv_lstm2d/while/clip_by_value_1/MinimumMinimumconv_lstm2d/while/Add_3:z:04conv_lstm2d/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ h
#conv_lstm2d/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!conv_lstm2d/while/clip_by_value_1Maximum-conv_lstm2d/while/clip_by_value_1/Minimum:z:0,conv_lstm2d/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@  
conv_lstm2d/while/mul_2Mul%conv_lstm2d/while/clip_by_value_1:z:0conv_lstm2d_while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
conv_lstm2d/while/add_4AddV2$conv_lstm2d/while/BiasAdd_2:output:0(conv_lstm2d/while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
conv_lstm2d/while/ReluReluconv_lstm2d/while/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
conv_lstm2d/while/mul_3Mul#conv_lstm2d/while/clip_by_value:z:0$conv_lstm2d/while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/add_5AddV2conv_lstm2d/while/mul_2:z:0conv_lstm2d/while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
conv_lstm2d/while/add_6AddV2$conv_lstm2d/while/BiasAdd_3:output:0(conv_lstm2d/while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
conv_lstm2d/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
conv_lstm2d/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/while/Mul_4Mulconv_lstm2d/while/add_6:z:0"conv_lstm2d/while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/Add_7AddV2conv_lstm2d/while/Mul_4:z:0"conv_lstm2d/while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
+conv_lstm2d/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
)conv_lstm2d/while/clip_by_value_2/MinimumMinimumconv_lstm2d/while/Add_7:z:04conv_lstm2d/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ h
#conv_lstm2d/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!conv_lstm2d/while/clip_by_value_2Maximum-conv_lstm2d/while/clip_by_value_2/Minimum:z:0,conv_lstm2d/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
conv_lstm2d/while/Relu_1Reluconv_lstm2d/while/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §
conv_lstm2d/while/mul_5Mul%conv_lstm2d/while/clip_by_value_2:z:0&conv_lstm2d/while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ è
6conv_lstm2d/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemconv_lstm2d_while_placeholder_1conv_lstm2d_while_placeholderconv_lstm2d/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒ[
conv_lstm2d/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d/while/add_8AddV2conv_lstm2d_while_placeholder"conv_lstm2d/while/add_8/y:output:0*
T0*
_output_shapes
: [
conv_lstm2d/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d/while/add_9AddV20conv_lstm2d_while_conv_lstm2d_while_loop_counter"conv_lstm2d/while/add_9/y:output:0*
T0*
_output_shapes
: }
conv_lstm2d/while/IdentityIdentityconv_lstm2d/while/add_9:z:0^conv_lstm2d/while/NoOp*
T0*
_output_shapes
: 
conv_lstm2d/while/Identity_1Identity6conv_lstm2d_while_conv_lstm2d_while_maximum_iterations^conv_lstm2d/while/NoOp*
T0*
_output_shapes
: 
conv_lstm2d/while/Identity_2Identityconv_lstm2d/while/add_8:z:0^conv_lstm2d/while/NoOp*
T0*
_output_shapes
: ½
conv_lstm2d/while/Identity_3IdentityFconv_lstm2d/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lstm2d/while/NoOp*
T0*
_output_shapes
: :éèÒ
conv_lstm2d/while/Identity_4Identityconv_lstm2d/while/mul_5:z:0^conv_lstm2d/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/Identity_5Identityconv_lstm2d/while/add_5:z:0^conv_lstm2d/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ×
conv_lstm2d/while/NoOpNoOp'^conv_lstm2d/while/split/ReadVariableOp)^conv_lstm2d/while/split_1/ReadVariableOp)^conv_lstm2d/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "\
+conv_lstm2d_while_conv_lstm2d_strided_slice-conv_lstm2d_while_conv_lstm2d_strided_slice_0"A
conv_lstm2d_while_identity#conv_lstm2d/while/Identity:output:0"E
conv_lstm2d_while_identity_1%conv_lstm2d/while/Identity_1:output:0"E
conv_lstm2d_while_identity_2%conv_lstm2d/while/Identity_2:output:0"E
conv_lstm2d_while_identity_3%conv_lstm2d/while/Identity_3:output:0"E
conv_lstm2d_while_identity_4%conv_lstm2d/while/Identity_4:output:0"E
conv_lstm2d_while_identity_5%conv_lstm2d/while/Identity_5:output:0"h
1conv_lstm2d_while_split_1_readvariableop_resource3conv_lstm2d_while_split_1_readvariableop_resource_0"h
1conv_lstm2d_while_split_2_readvariableop_resource3conv_lstm2d_while_split_2_readvariableop_resource_0"d
/conv_lstm2d_while_split_readvariableop_resource1conv_lstm2d_while_split_readvariableop_resource_0"Ø
iconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensorkconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 2P
&conv_lstm2d/while/split/ReadVariableOp&conv_lstm2d/while/split/ReadVariableOp2T
(conv_lstm2d/while/split_1/ReadVariableOp(conv_lstm2d/while/split_1/ReadVariableOp2T
(conv_lstm2d/while/split_2/ReadVariableOp(conv_lstm2d/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
»
È
.__inference_conv_lstm2d_1_layer_call_fn_173336

inputs!
unknown:@#
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_170428{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
 
_user_specified_nameinputs
ºp
ë

conv_lstm2d_while_body_1718874
0conv_lstm2d_while_conv_lstm2d_while_loop_counter:
6conv_lstm2d_while_conv_lstm2d_while_maximum_iterations!
conv_lstm2d_while_placeholder#
conv_lstm2d_while_placeholder_1#
conv_lstm2d_while_placeholder_2#
conv_lstm2d_while_placeholder_31
-conv_lstm2d_while_conv_lstm2d_strided_slice_0o
kconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0K
1conv_lstm2d_while_split_readvariableop_resource_0: M
3conv_lstm2d_while_split_1_readvariableop_resource_0: A
3conv_lstm2d_while_split_2_readvariableop_resource_0: 
conv_lstm2d_while_identity 
conv_lstm2d_while_identity_1 
conv_lstm2d_while_identity_2 
conv_lstm2d_while_identity_3 
conv_lstm2d_while_identity_4 
conv_lstm2d_while_identity_5/
+conv_lstm2d_while_conv_lstm2d_strided_slicem
iconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensorI
/conv_lstm2d_while_split_readvariableop_resource: K
1conv_lstm2d_while_split_1_readvariableop_resource: ?
1conv_lstm2d_while_split_2_readvariableop_resource: ¢&conv_lstm2d/while/split/ReadVariableOp¢(conv_lstm2d/while/split_1/ReadVariableOp¢(conv_lstm2d/while/split_2/ReadVariableOp
Cconv_lstm2d/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ê
5conv_lstm2d/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0conv_lstm2d_while_placeholderLconv_lstm2d/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0c
!conv_lstm2d/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&conv_lstm2d/while/split/ReadVariableOpReadVariableOp1conv_lstm2d_while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0ô
conv_lstm2d/while/splitSplit*conv_lstm2d/while/split/split_dim:output:0.conv_lstm2d/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splite
#conv_lstm2d/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¤
(conv_lstm2d/while/split_1/ReadVariableOpReadVariableOp3conv_lstm2d_while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0ú
conv_lstm2d/while/split_1Split,conv_lstm2d/while/split_1/split_dim:output:00conv_lstm2d/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splite
#conv_lstm2d/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(conv_lstm2d/while/split_2/ReadVariableOpReadVariableOp3conv_lstm2d_while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0Ê
conv_lstm2d/while/split_2Split,conv_lstm2d/while/split_2/split_dim:output:00conv_lstm2d/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splité
conv_lstm2d/while/convolutionConv2D<conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0 conv_lstm2d/while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
ª
conv_lstm2d/while/BiasAddBiasAdd&conv_lstm2d/while/convolution:output:0"conv_lstm2d/while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ë
conv_lstm2d/while/convolution_1Conv2D<conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0 conv_lstm2d/while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
conv_lstm2d/while/BiasAdd_1BiasAdd(conv_lstm2d/while/convolution_1:output:0"conv_lstm2d/while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ë
conv_lstm2d/while/convolution_2Conv2D<conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0 conv_lstm2d/while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
conv_lstm2d/while/BiasAdd_2BiasAdd(conv_lstm2d/while/convolution_2:output:0"conv_lstm2d/while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ë
conv_lstm2d/while/convolution_3Conv2D<conv_lstm2d/while/TensorArrayV2Read/TensorListGetItem:item:0 conv_lstm2d/while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
®
conv_lstm2d/while/BiasAdd_3BiasAdd(conv_lstm2d/while/convolution_3:output:0"conv_lstm2d/while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ð
conv_lstm2d/while/convolution_4Conv2Dconv_lstm2d_while_placeholder_2"conv_lstm2d/while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Ð
conv_lstm2d/while/convolution_5Conv2Dconv_lstm2d_while_placeholder_2"conv_lstm2d/while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Ð
conv_lstm2d/while/convolution_6Conv2Dconv_lstm2d_while_placeholder_2"conv_lstm2d/while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Ð
conv_lstm2d/while/convolution_7Conv2Dconv_lstm2d_while_placeholder_2"conv_lstm2d/while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¦
conv_lstm2d/while/addAddV2"conv_lstm2d/while/BiasAdd:output:0(conv_lstm2d/while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
conv_lstm2d/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
conv_lstm2d/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/while/MulMulconv_lstm2d/while/add:z:0 conv_lstm2d/while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/Add_1AddV2conv_lstm2d/while/Mul:z:0"conv_lstm2d/while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ n
)conv_lstm2d/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?½
'conv_lstm2d/while/clip_by_value/MinimumMinimumconv_lstm2d/while/Add_1:z:02conv_lstm2d/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ f
!conv_lstm2d/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
conv_lstm2d/while/clip_by_valueMaximum+conv_lstm2d/while/clip_by_value/Minimum:z:0*conv_lstm2d/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
conv_lstm2d/while/add_2AddV2$conv_lstm2d/while/BiasAdd_1:output:0(conv_lstm2d/while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
conv_lstm2d/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
conv_lstm2d/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/while/Mul_1Mulconv_lstm2d/while/add_2:z:0"conv_lstm2d/while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/Add_3AddV2conv_lstm2d/while/Mul_1:z:0"conv_lstm2d/while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
+conv_lstm2d/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
)conv_lstm2d/while/clip_by_value_1/MinimumMinimumconv_lstm2d/while/Add_3:z:04conv_lstm2d/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ h
#conv_lstm2d/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!conv_lstm2d/while/clip_by_value_1Maximum-conv_lstm2d/while/clip_by_value_1/Minimum:z:0,conv_lstm2d/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@  
conv_lstm2d/while/mul_2Mul%conv_lstm2d/while/clip_by_value_1:z:0conv_lstm2d_while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
conv_lstm2d/while/add_4AddV2$conv_lstm2d/while/BiasAdd_2:output:0(conv_lstm2d/while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
conv_lstm2d/while/ReluReluconv_lstm2d/while/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
conv_lstm2d/while/mul_3Mul#conv_lstm2d/while/clip_by_value:z:0$conv_lstm2d/while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/add_5AddV2conv_lstm2d/while/mul_2:z:0conv_lstm2d/while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ª
conv_lstm2d/while/add_6AddV2$conv_lstm2d/while/BiasAdd_3:output:0(conv_lstm2d/while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
conv_lstm2d/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>^
conv_lstm2d/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/while/Mul_4Mulconv_lstm2d/while/add_6:z:0"conv_lstm2d/while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/Add_7AddV2conv_lstm2d/while/Mul_4:z:0"conv_lstm2d/while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
+conv_lstm2d/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
)conv_lstm2d/while/clip_by_value_2/MinimumMinimumconv_lstm2d/while/Add_7:z:04conv_lstm2d/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ h
#conv_lstm2d/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
!conv_lstm2d/while/clip_by_value_2Maximum-conv_lstm2d/while/clip_by_value_2/Minimum:z:0,conv_lstm2d/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
conv_lstm2d/while/Relu_1Reluconv_lstm2d/while/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §
conv_lstm2d/while/mul_5Mul%conv_lstm2d/while/clip_by_value_2:z:0&conv_lstm2d/while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ è
6conv_lstm2d/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemconv_lstm2d_while_placeholder_1conv_lstm2d_while_placeholderconv_lstm2d/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒ[
conv_lstm2d/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d/while/add_8AddV2conv_lstm2d_while_placeholder"conv_lstm2d/while/add_8/y:output:0*
T0*
_output_shapes
: [
conv_lstm2d/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d/while/add_9AddV20conv_lstm2d_while_conv_lstm2d_while_loop_counter"conv_lstm2d/while/add_9/y:output:0*
T0*
_output_shapes
: }
conv_lstm2d/while/IdentityIdentityconv_lstm2d/while/add_9:z:0^conv_lstm2d/while/NoOp*
T0*
_output_shapes
: 
conv_lstm2d/while/Identity_1Identity6conv_lstm2d_while_conv_lstm2d_while_maximum_iterations^conv_lstm2d/while/NoOp*
T0*
_output_shapes
: 
conv_lstm2d/while/Identity_2Identityconv_lstm2d/while/add_8:z:0^conv_lstm2d/while/NoOp*
T0*
_output_shapes
: ½
conv_lstm2d/while/Identity_3IdentityFconv_lstm2d/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lstm2d/while/NoOp*
T0*
_output_shapes
: :éèÒ
conv_lstm2d/while/Identity_4Identityconv_lstm2d/while/mul_5:z:0^conv_lstm2d/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/while/Identity_5Identityconv_lstm2d/while/add_5:z:0^conv_lstm2d/while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ×
conv_lstm2d/while/NoOpNoOp'^conv_lstm2d/while/split/ReadVariableOp)^conv_lstm2d/while/split_1/ReadVariableOp)^conv_lstm2d/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "\
+conv_lstm2d_while_conv_lstm2d_strided_slice-conv_lstm2d_while_conv_lstm2d_strided_slice_0"A
conv_lstm2d_while_identity#conv_lstm2d/while/Identity:output:0"E
conv_lstm2d_while_identity_1%conv_lstm2d/while/Identity_1:output:0"E
conv_lstm2d_while_identity_2%conv_lstm2d/while/Identity_2:output:0"E
conv_lstm2d_while_identity_3%conv_lstm2d/while/Identity_3:output:0"E
conv_lstm2d_while_identity_4%conv_lstm2d/while/Identity_4:output:0"E
conv_lstm2d_while_identity_5%conv_lstm2d/while/Identity_5:output:0"h
1conv_lstm2d_while_split_1_readvariableop_resource3conv_lstm2d_while_split_1_readvariableop_resource_0"h
1conv_lstm2d_while_split_2_readvariableop_resource3conv_lstm2d_while_split_2_readvariableop_resource_0"d
/conv_lstm2d_while_split_readvariableop_resource1conv_lstm2d_while_split_readvariableop_resource_0"Ø
iconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensorkconv_lstm2d_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 2P
&conv_lstm2d/while/split/ReadVariableOp&conv_lstm2d/while/split/ReadVariableOp2T
(conv_lstm2d/while/split_1/ReadVariableOp(conv_lstm2d/while/split_1/ReadVariableOp2T
(conv_lstm2d/while/split_2/ReadVariableOp(conv_lstm2d/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
½!

while_body_169119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_169143_0: (
while_169145_0: 
while_169147_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_169143: &
while_169145: 
while_169147: ¢while/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_169143_0while_169145_0while_169147_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_169105Ï
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ l

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
while_169143while_169143_0"
while_169145while_169145_0"
while_169147while_169147_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
Z
ë
while_body_172911
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0: A
'while_split_1_readvariableop_resource_0: 5
'while_split_2_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource: ?
%while_split_1_readvariableop_resource: 3
%while_split_2_readvariableop_resource: ¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
Ñ
Á
while_cond_170669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_170669___redundant_placeholder04
0while_while_cond_170669___redundant_placeholder14
0while_while_cond_170669___redundant_placeholder24
0while_while_cond_170669___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
°b
Ú
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_171038

inputs7
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢while]

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                u
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_170912*
condR
while_cond_170911*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ê
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ j
IdentityIdentitytranspose_1:y:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
³1
é
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_169412

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
identity¢StatefulPartitionedCall¢whilef

zeros_like	ZerosLikeinputs*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ~
	transpose	Transposeinputstranspose/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskï
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_169293v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_169344*
condR
while_cond_169343*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ó
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                §
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ s
IdentityIdentitytranspose_1:y:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ h
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ : : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs
Z
ë
while_body_172471
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0: A
'while_split_1_readvariableop_resource_0: 5
'while_split_2_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource: ?
%while_split_1_readvariableop_resource: 3
%while_split_2_readvariableop_resource: ¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
ö

 
&__inference_model_layer_call_fn_170541
input_1!
unknown: #
	unknown_0: 
	unknown_1: #
	unknown_2:@#
	unknown_3:@
	unknown_4:@#
	unknown_5:
	unknown_6:#
	unknown_7:

	unknown_8:

identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_170518{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
!
_user_specified_name	input_1
ß

/__inference_conv_lstm_cell_layer_call_fn_174405

inputs
states_0
states_1!
unknown: #
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_169293w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
 
_user_specified_nameinputs:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
states/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
states/1
3
ë
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_169894

inputs!
unknown:@#
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCall¢whilef

zeros_like	ZerosLikeinputs*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    t
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ~
	transpose	Transposeinputstranspose/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskñ
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_169773v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_169826*
condR
while_cond_169825*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          Ó
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                §
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
IdentityIdentitytranspose_1:y:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ h
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Z
ë
while_body_170912
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0: A
'while_split_1_readvariableop_resource_0: 5
'while_split_2_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource: ?
%while_split_1_readvariableop_resource: 3
%while_split_2_readvariableop_resource: ¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
¨%
±
D__inference_conv2d_1_layer_call_and_return_conditional_losses_174371

inputs?
%conv2d_conv2d_readvariableop_resource:
@
2squeeze_batch_dims_biasadd_readvariableop_resource:

identity¢Conv2D/Conv2D/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0¸
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@       
   ¥
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 

)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0·
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:®
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
w
SoftmaxSoftmax%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
l
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 

NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd@ : : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
ÐR
ô
__inference__traced_save_174906
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop1
-savev2_conv_lstm2d_kernel_read_readvariableop;
7savev2_conv_lstm2d_recurrent_kernel_read_readvariableop/
+savev2_conv_lstm2d_bias_read_readvariableop3
/savev2_conv_lstm2d_1_kernel_read_readvariableop=
9savev2_conv_lstm2d_1_recurrent_kernel_read_readvariableop1
-savev2_conv_lstm2d_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop8
4savev2_adam_conv_lstm2d_kernel_m_read_readvariableopB
>savev2_adam_conv_lstm2d_recurrent_kernel_m_read_readvariableop6
2savev2_adam_conv_lstm2d_bias_m_read_readvariableop:
6savev2_adam_conv_lstm2d_1_kernel_m_read_readvariableopD
@savev2_adam_conv_lstm2d_1_recurrent_kernel_m_read_readvariableop8
4savev2_adam_conv_lstm2d_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop8
4savev2_adam_conv_lstm2d_kernel_v_read_readvariableopB
>savev2_adam_conv_lstm2d_recurrent_kernel_v_read_readvariableop6
2savev2_adam_conv_lstm2d_bias_v_read_readvariableop:
6savev2_adam_conv_lstm2d_1_kernel_v_read_readvariableopD
@savev2_adam_conv_lstm2d_1_recurrent_kernel_v_read_readvariableop8
4savev2_adam_conv_lstm2d_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Û
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueúB÷(B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH½
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ã
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop-savev2_conv_lstm2d_kernel_read_readvariableop7savev2_conv_lstm2d_recurrent_kernel_read_readvariableop+savev2_conv_lstm2d_bias_read_readvariableop/savev2_conv_lstm2d_1_kernel_read_readvariableop9savev2_conv_lstm2d_1_recurrent_kernel_read_readvariableop-savev2_conv_lstm2d_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop4savev2_adam_conv_lstm2d_kernel_m_read_readvariableop>savev2_adam_conv_lstm2d_recurrent_kernel_m_read_readvariableop2savev2_adam_conv_lstm2d_bias_m_read_readvariableop6savev2_adam_conv_lstm2d_1_kernel_m_read_readvariableop@savev2_adam_conv_lstm2d_1_recurrent_kernel_m_read_readvariableop4savev2_adam_conv_lstm2d_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop4savev2_adam_conv_lstm2d_kernel_v_read_readvariableop>savev2_adam_conv_lstm2d_recurrent_kernel_v_read_readvariableop2savev2_adam_conv_lstm2d_bias_v_read_readvariableop6savev2_adam_conv_lstm2d_1_kernel_v_read_readvariableop@savev2_adam_conv_lstm2d_1_recurrent_kernel_v_read_readvariableop4savev2_adam_conv_lstm2d_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*·
_input_shapes¥
¢: :::
:
: : : : : : : : :@:@:@: : : : :::
:
: : : :@:@:@:::
:
: : : :@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :,
(
&
_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@:
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
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:
: !

_output_shapes
:
:,"(
&
_output_shapes
: :,#(
&
_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
:@:,&(
&
_output_shapes
:@: '

_output_shapes
:@:(

_output_shapes
: 
¯Ò
Ú	
A__inference_model_layer_call_and_return_conditional_losses_171793

inputsC
)conv_lstm2d_split_readvariableop_resource: E
+conv_lstm2d_split_1_readvariableop_resource: 9
+conv_lstm2d_split_2_readvariableop_resource: E
+conv_lstm2d_1_split_readvariableop_resource:@G
-conv_lstm2d_1_split_1_readvariableop_resource:@;
-conv_lstm2d_1_split_2_readvariableop_resource:@F
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:
I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:

identity¢#conv2d/Conv2D/Conv2D/ReadVariableOp¢0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_1/Conv2D/Conv2D/ReadVariableOp¢2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp¢ conv_lstm2d/split/ReadVariableOp¢"conv_lstm2d/split_1/ReadVariableOp¢"conv_lstm2d/split_2/ReadVariableOp¢conv_lstm2d/while¢"conv_lstm2d_1/split/ReadVariableOp¢$conv_lstm2d_1/split_1/ReadVariableOp¢$conv_lstm2d_1/split_2/ReadVariableOp¢conv_lstm2d_1/whilei
conv_lstm2d/zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ c
!conv_lstm2d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d/SumSumconv_lstm2d/zeros_like:y:0*conv_lstm2d/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
conv_lstm2d/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    ¹
conv_lstm2d/convolutionConv2Dconv_lstm2d/Sum:output:0conv_lstm2d/zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
w
conv_lstm2d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
conv_lstm2d/transpose	Transposeinputs#conv_lstm2d/transpose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ Z
conv_lstm2d/ShapeShapeconv_lstm2d/transpose:y:0*
T0*
_output_shapes
:i
conv_lstm2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!conv_lstm2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!conv_lstm2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv_lstm2d/strided_sliceStridedSliceconv_lstm2d/Shape:output:0(conv_lstm2d/strided_slice/stack:output:0*conv_lstm2d/strided_slice/stack_1:output:0*conv_lstm2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'conv_lstm2d/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
conv_lstm2d/TensorArrayV2TensorListReserve0conv_lstm2d/TensorArrayV2/element_shape:output:0"conv_lstm2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aconv_lstm2d/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          
3conv_lstm2d/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lstm2d/transpose:y:0Jconv_lstm2d/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
!conv_lstm2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv_lstm2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv_lstm2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
conv_lstm2d/strided_slice_1StridedSliceconv_lstm2d/transpose:y:0*conv_lstm2d/strided_slice_1/stack:output:0,conv_lstm2d/strided_slice_1/stack_1:output:0,conv_lstm2d/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_mask]
conv_lstm2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 conv_lstm2d/split/ReadVariableOpReadVariableOp)conv_lstm2d_split_readvariableop_resource*&
_output_shapes
: *
dtype0â
conv_lstm2d/splitSplit$conv_lstm2d/split/split_dim:output:0(conv_lstm2d/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split_
conv_lstm2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
"conv_lstm2d/split_1/ReadVariableOpReadVariableOp+conv_lstm2d_split_1_readvariableop_resource*&
_output_shapes
: *
dtype0è
conv_lstm2d/split_1Split&conv_lstm2d/split_1/split_dim:output:0*conv_lstm2d/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split_
conv_lstm2d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"conv_lstm2d/split_2/ReadVariableOpReadVariableOp+conv_lstm2d_split_2_readvariableop_resource*
_output_shapes
: *
dtype0¸
conv_lstm2d/split_2Split&conv_lstm2d/split_2/split_dim:output:0*conv_lstm2d/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÇ
conv_lstm2d/convolution_1Conv2D$conv_lstm2d/strided_slice_1:output:0conv_lstm2d/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/BiasAddBiasAdd"conv_lstm2d/convolution_1:output:0conv_lstm2d/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
conv_lstm2d/convolution_2Conv2D$conv_lstm2d/strided_slice_1:output:0conv_lstm2d/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/BiasAdd_1BiasAdd"conv_lstm2d/convolution_2:output:0conv_lstm2d/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
conv_lstm2d/convolution_3Conv2D$conv_lstm2d/strided_slice_1:output:0conv_lstm2d/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/BiasAdd_2BiasAdd"conv_lstm2d/convolution_3:output:0conv_lstm2d/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
conv_lstm2d/convolution_4Conv2D$conv_lstm2d/strided_slice_1:output:0conv_lstm2d/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/BiasAdd_3BiasAdd"conv_lstm2d/convolution_4:output:0conv_lstm2d/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Å
conv_lstm2d/convolution_5Conv2D conv_lstm2d/convolution:output:0conv_lstm2d/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Å
conv_lstm2d/convolution_6Conv2D conv_lstm2d/convolution:output:0conv_lstm2d/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Å
conv_lstm2d/convolution_7Conv2D conv_lstm2d/convolution:output:0conv_lstm2d/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
Å
conv_lstm2d/convolution_8Conv2D conv_lstm2d/convolution:output:0conv_lstm2d/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

conv_lstm2d/addAddV2conv_lstm2d/BiasAdd:output:0"conv_lstm2d/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
conv_lstm2d/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>X
conv_lstm2d/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/MulMulconv_lstm2d/add:z:0conv_lstm2d/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/Add_1AddV2conv_lstm2d/Mul:z:0conv_lstm2d/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ h
#conv_lstm2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
!conv_lstm2d/clip_by_value/MinimumMinimumconv_lstm2d/Add_1:z:0,conv_lstm2d/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ `
conv_lstm2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    «
conv_lstm2d/clip_by_valueMaximum%conv_lstm2d/clip_by_value/Minimum:z:0$conv_lstm2d/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/add_2AddV2conv_lstm2d/BiasAdd_1:output:0"conv_lstm2d/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ X
conv_lstm2d/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>X
conv_lstm2d/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/Mul_1Mulconv_lstm2d/add_2:z:0conv_lstm2d/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/Add_3AddV2conv_lstm2d/Mul_1:z:0conv_lstm2d/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
%conv_lstm2d/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
#conv_lstm2d/clip_by_value_1/MinimumMinimumconv_lstm2d/Add_3:z:0.conv_lstm2d/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
conv_lstm2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
conv_lstm2d/clip_by_value_1Maximum'conv_lstm2d/clip_by_value_1/Minimum:z:0&conv_lstm2d/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/mul_2Mulconv_lstm2d/clip_by_value_1:z:0 conv_lstm2d/convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/add_4AddV2conv_lstm2d/BiasAdd_2:output:0"conv_lstm2d/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ i
conv_lstm2d/ReluReluconv_lstm2d/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/mul_3Mulconv_lstm2d/clip_by_value:z:0conv_lstm2d/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/add_5AddV2conv_lstm2d/mul_2:z:0conv_lstm2d/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/add_6AddV2conv_lstm2d/BiasAdd_3:output:0"conv_lstm2d/convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ X
conv_lstm2d/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>X
conv_lstm2d/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d/Mul_4Mulconv_lstm2d/add_6:z:0conv_lstm2d/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/Add_7AddV2conv_lstm2d/Mul_4:z:0conv_lstm2d/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
%conv_lstm2d/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
#conv_lstm2d/clip_by_value_2/MinimumMinimumconv_lstm2d/Add_7:z:0.conv_lstm2d/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
conv_lstm2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
conv_lstm2d/clip_by_value_2Maximum'conv_lstm2d/clip_by_value_2/Minimum:z:0&conv_lstm2d/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ k
conv_lstm2d/Relu_1Reluconv_lstm2d/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d/mul_5Mulconv_lstm2d/clip_by_value_2:z:0 conv_lstm2d/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
)conv_lstm2d/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ú
conv_lstm2d/TensorArrayV2_1TensorListReserve2conv_lstm2d/TensorArrayV2_1/element_shape:output:0"conv_lstm2d/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒR
conv_lstm2d/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$conv_lstm2d/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ`
conv_lstm2d/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©
conv_lstm2d/whileWhile'conv_lstm2d/while/loop_counter:output:0-conv_lstm2d/while/maximum_iterations:output:0conv_lstm2d/time:output:0$conv_lstm2d/TensorArrayV2_1:handle:0 conv_lstm2d/convolution:output:0 conv_lstm2d/convolution:output:0"conv_lstm2d/strided_slice:output:0Cconv_lstm2d/TensorArrayUnstack/TensorListFromTensor:output_handle:0)conv_lstm2d_split_readvariableop_resource+conv_lstm2d_split_1_readvariableop_resource+conv_lstm2d_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
conv_lstm2d_while_body_171374*)
cond!R
conv_lstm2d_while_cond_171373*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
<conv_lstm2d/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          î
.conv_lstm2d/TensorArrayV2Stack/TensorListStackTensorListStackconv_lstm2d/while:output:3Econv_lstm2d/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ *
element_dtype0t
!conv_lstm2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#conv_lstm2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#conv_lstm2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
conv_lstm2d/strided_slice_2StridedSlice7conv_lstm2d/TensorArrayV2Stack/TensorListStack:tensor:0*conv_lstm2d/strided_slice_2/stack:output:0,conv_lstm2d/strided_slice_2/stack_1:output:0,conv_lstm2d/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_masky
conv_lstm2d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                Â
conv_lstm2d/transpose_1	Transpose7conv_lstm2d/TensorArrayV2Stack/TensorListStack:tensor:0%conv_lstm2d/transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          £
time_distributed/ReshapeReshapeconv_lstm2d/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Á
&time_distributed/max_pooling2d/MaxPoolMaxPool!time_distributed/Reshape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
}
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿd             ¿
time_distributed/Reshape_1Reshape/time_distributed/max_pooling2d/MaxPool:output:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          §
time_distributed/Reshape_2Reshapeconv_lstm2d/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
conv_lstm2d_1/zeros_like	ZerosLike#time_distributed/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd e
#conv_lstm2d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
conv_lstm2d_1/SumSumconv_lstm2d_1/zeros_like:y:0,conv_lstm2d_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
#conv_lstm2d_1/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            ^
conv_lstm2d_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
conv_lstm2d_1/zerosFill,conv_lstm2d_1/zeros/shape_as_tensor:output:0"conv_lstm2d_1/zeros/Const:output:0*
T0*&
_output_shapes
:¿
conv_lstm2d_1/convolutionConv2Dconv_lstm2d_1/Sum:output:0conv_lstm2d_1/zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
y
conv_lstm2d_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ®
conv_lstm2d_1/transpose	Transpose#time_distributed/Reshape_1:output:0%conv_lstm2d_1/transpose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ ^
conv_lstm2d_1/ShapeShapeconv_lstm2d_1/transpose:y:0*
T0*
_output_shapes
:k
!conv_lstm2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv_lstm2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv_lstm2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv_lstm2d_1/strided_sliceStridedSliceconv_lstm2d_1/Shape:output:0*conv_lstm2d_1/strided_slice/stack:output:0,conv_lstm2d_1/strided_slice/stack_1:output:0,conv_lstm2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)conv_lstm2d_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÜ
conv_lstm2d_1/TensorArrayV2TensorListReserve2conv_lstm2d_1/TensorArrayV2/element_shape:output:0$conv_lstm2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Cconv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          
5conv_lstm2d_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lstm2d_1/transpose:y:0Lconv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#conv_lstm2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%conv_lstm2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%conv_lstm2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:·
conv_lstm2d_1/strided_slice_1StridedSliceconv_lstm2d_1/transpose:y:0,conv_lstm2d_1/strided_slice_1/stack:output:0.conv_lstm2d_1/strided_slice_1/stack_1:output:0.conv_lstm2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask_
conv_lstm2d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
"conv_lstm2d_1/split/ReadVariableOpReadVariableOp+conv_lstm2d_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype0è
conv_lstm2d_1/splitSplit&conv_lstm2d_1/split/split_dim:output:0*conv_lstm2d_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splita
conv_lstm2d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$conv_lstm2d_1/split_1/ReadVariableOpReadVariableOp-conv_lstm2d_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype0î
conv_lstm2d_1/split_1Split(conv_lstm2d_1/split_1/split_dim:output:0,conv_lstm2d_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splita
conv_lstm2d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
$conv_lstm2d_1/split_2/ReadVariableOpReadVariableOp-conv_lstm2d_1_split_2_readvariableop_resource*
_output_shapes
:@*
dtype0¾
conv_lstm2d_1/split_2Split(conv_lstm2d_1/split_2/split_dim:output:0,conv_lstm2d_1/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÍ
conv_lstm2d_1/convolution_1Conv2D&conv_lstm2d_1/strided_slice_1:output:0conv_lstm2d_1/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
 
conv_lstm2d_1/BiasAddBiasAdd$conv_lstm2d_1/convolution_1:output:0conv_lstm2d_1/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Í
conv_lstm2d_1/convolution_2Conv2D&conv_lstm2d_1/strided_slice_1:output:0conv_lstm2d_1/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¢
conv_lstm2d_1/BiasAdd_1BiasAdd$conv_lstm2d_1/convolution_2:output:0conv_lstm2d_1/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Í
conv_lstm2d_1/convolution_3Conv2D&conv_lstm2d_1/strided_slice_1:output:0conv_lstm2d_1/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¢
conv_lstm2d_1/BiasAdd_2BiasAdd$conv_lstm2d_1/convolution_3:output:0conv_lstm2d_1/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Í
conv_lstm2d_1/convolution_4Conv2D&conv_lstm2d_1/strided_slice_1:output:0conv_lstm2d_1/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¢
conv_lstm2d_1/BiasAdd_3BiasAdd$conv_lstm2d_1/convolution_4:output:0conv_lstm2d_1/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ë
conv_lstm2d_1/convolution_5Conv2D"conv_lstm2d_1/convolution:output:0conv_lstm2d_1/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ë
conv_lstm2d_1/convolution_6Conv2D"conv_lstm2d_1/convolution:output:0conv_lstm2d_1/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ë
conv_lstm2d_1/convolution_7Conv2D"conv_lstm2d_1/convolution:output:0conv_lstm2d_1/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
Ë
conv_lstm2d_1/convolution_8Conv2D"conv_lstm2d_1/convolution:output:0conv_lstm2d_1/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv_lstm2d_1/addAddV2conv_lstm2d_1/BiasAdd:output:0$conv_lstm2d_1/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
conv_lstm2d_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Z
conv_lstm2d_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/MulMulconv_lstm2d_1/add:z:0conv_lstm2d_1/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/Add_1AddV2conv_lstm2d_1/Mul:z:0conv_lstm2d_1/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
%conv_lstm2d_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
#conv_lstm2d_1/clip_by_value/MinimumMinimumconv_lstm2d_1/Add_1:z:0.conv_lstm2d_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
conv_lstm2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
conv_lstm2d_1/clip_by_valueMaximum'conv_lstm2d_1/clip_by_value/Minimum:z:0&conv_lstm2d_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/add_2AddV2 conv_lstm2d_1/BiasAdd_1:output:0$conv_lstm2d_1/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
conv_lstm2d_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Z
conv_lstm2d_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/Mul_1Mulconv_lstm2d_1/add_2:z:0conv_lstm2d_1/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/Add_3AddV2conv_lstm2d_1/Mul_1:z:0conv_lstm2d_1/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
'conv_lstm2d_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
%conv_lstm2d_1/clip_by_value_1/MinimumMinimumconv_lstm2d_1/Add_3:z:00conv_lstm2d_1/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
conv_lstm2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
conv_lstm2d_1/clip_by_value_1Maximum)conv_lstm2d_1/clip_by_value_1/Minimum:z:0(conv_lstm2d_1/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/mul_2Mul!conv_lstm2d_1/clip_by_value_1:z:0"conv_lstm2d_1/convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/add_4AddV2 conv_lstm2d_1/BiasAdd_2:output:0$conv_lstm2d_1/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
conv_lstm2d_1/ReluReluconv_lstm2d_1/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/mul_3Mulconv_lstm2d_1/clip_by_value:z:0 conv_lstm2d_1/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/add_5AddV2conv_lstm2d_1/mul_2:z:0conv_lstm2d_1/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/add_6AddV2 conv_lstm2d_1/BiasAdd_3:output:0$conv_lstm2d_1/convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
conv_lstm2d_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Z
conv_lstm2d_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
conv_lstm2d_1/Mul_4Mulconv_lstm2d_1/add_6:z:0conv_lstm2d_1/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/Add_7AddV2conv_lstm2d_1/Mul_4:z:0conv_lstm2d_1/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
'conv_lstm2d_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
%conv_lstm2d_1/clip_by_value_2/MinimumMinimumconv_lstm2d_1/Add_7:z:00conv_lstm2d_1/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
conv_lstm2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
conv_lstm2d_1/clip_by_value_2Maximum)conv_lstm2d_1/clip_by_value_2/Minimum:z:0(conv_lstm2d_1/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
conv_lstm2d_1/Relu_1Reluconv_lstm2d_1/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_lstm2d_1/mul_5Mul!conv_lstm2d_1/clip_by_value_2:z:0"conv_lstm2d_1/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+conv_lstm2d_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
conv_lstm2d_1/TensorArrayV2_1TensorListReserve4conv_lstm2d_1/TensorArrayV2_1/element_shape:output:0$conv_lstm2d_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
conv_lstm2d_1/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&conv_lstm2d_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 conv_lstm2d_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Å
conv_lstm2d_1/whileWhile)conv_lstm2d_1/while/loop_counter:output:0/conv_lstm2d_1/while/maximum_iterations:output:0conv_lstm2d_1/time:output:0&conv_lstm2d_1/TensorArrayV2_1:handle:0"conv_lstm2d_1/convolution:output:0"conv_lstm2d_1/convolution:output:0$conv_lstm2d_1/strided_slice:output:0Econv_lstm2d_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0+conv_lstm2d_1_split_readvariableop_resource-conv_lstm2d_1_split_1_readvariableop_resource-conv_lstm2d_1_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
conv_lstm2d_1_while_body_171599*+
cond#R!
conv_lstm2d_1_while_cond_171598*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
>conv_lstm2d_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ô
0conv_lstm2d_1/TensorArrayV2Stack/TensorListStackTensorListStackconv_lstm2d_1/while:output:3Gconv_lstm2d_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ *
element_dtype0v
#conv_lstm2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%conv_lstm2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%conv_lstm2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Õ
conv_lstm2d_1/strided_slice_2StridedSlice9conv_lstm2d_1/TensorArrayV2Stack/TensorListStack:tensor:0,conv_lstm2d_1/strided_slice_2/stack:output:0.conv_lstm2d_1/strided_slice_2/stack_1:output:0.conv_lstm2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask{
conv_lstm2d_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                È
conv_lstm2d_1/transpose_1	Transpose9conv_lstm2d_1/TensorArrayV2Stack/TensorListStack:tensor:0'conv_lstm2d_1/transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ©
time_distributed_1/ReshapeReshapeconv_lstm2d_1/transpose_1:y:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
&time_distributed_1/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
(time_distributed_1/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ´
$time_distributed_1/up_sampling2d/mulMul/time_distributed_1/up_sampling2d/Const:output:01time_distributed_1/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:ù
=time_distributed_1/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#time_distributed_1/Reshape:output:0(time_distributed_1/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
half_pixel_centers(
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿd   @          â
time_distributed_1/Reshape_1ReshapeNtime_distributed_1/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ {
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ­
time_distributed_1/Reshape_2Reshapeconv_lstm2d_1/transpose_1:y:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
conv2d/Conv2D/ShapeShape%time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿm
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          §
conv2d/Conv2D/ReshapeReshape%time_distributed_1/Reshape_1:output:0$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Í
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿy
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          º
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¦
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ì
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@          p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Ã
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ ^
conv2d_1/Conv2D/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          
conv2d_1/Conv2D/ReshapeReshapeconv2d/Relu:activations:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0Ó
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@       
   À
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
ª
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ò
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 

+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 

conv2d_1/SoftmaxSoftmax.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
u
IdentityIdentityconv2d_1/Softmax:softmax:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 

NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp!^conv_lstm2d/split/ReadVariableOp#^conv_lstm2d/split_1/ReadVariableOp#^conv_lstm2d/split_2/ReadVariableOp^conv_lstm2d/while#^conv_lstm2d_1/split/ReadVariableOp%^conv_lstm2d_1/split_1/ReadVariableOp%^conv_lstm2d_1/split_2/ReadVariableOp^conv_lstm2d_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2D
 conv_lstm2d/split/ReadVariableOp conv_lstm2d/split/ReadVariableOp2H
"conv_lstm2d/split_1/ReadVariableOp"conv_lstm2d/split_1/ReadVariableOp2H
"conv_lstm2d/split_2/ReadVariableOp"conv_lstm2d/split_2/ReadVariableOp2&
conv_lstm2d/whileconv_lstm2d/while2H
"conv_lstm2d_1/split/ReadVariableOp"conv_lstm2d_1/split/ReadVariableOp2L
$conv_lstm2d_1/split_1/ReadVariableOp$conv_lstm2d_1/split_1/ReadVariableOp2L
$conv_lstm2d_1/split_2/ReadVariableOp$conv_lstm2d_1/split_2/ReadVariableOp2*
conv_lstm2d_1/whileconv_lstm2d_1/while:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
 $

A__inference_model_layer_call_and_return_conditional_losses_170518

inputs,
conv_lstm2d_170197: ,
conv_lstm2d_170199:  
conv_lstm2d_170201: .
conv_lstm2d_1_170429:@.
conv_lstm2d_1_170431:@"
conv_lstm2d_1_170433:@'
conv2d_170473:
conv2d_170475:)
conv2d_1_170512:

conv2d_1_170514:

identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢#conv_lstm2d/StatefulPartitionedCall¢%conv_lstm2d_1/StatefulPartitionedCall¡
#conv_lstm2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv_lstm2d_170197conv_lstm2d_170199conv_lstm2d_170201*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_170196ý
 time_distributed/PartitionedCallPartitionedCall,conv_lstm2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_169455w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ´
time_distributed/ReshapeReshape,conv_lstm2d/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Î
%conv_lstm2d_1/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0conv_lstm2d_1_170429conv_lstm2d_1_170431conv_lstm2d_1_170433*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_170428
"time_distributed_1/PartitionedCallPartitionedCall.conv_lstm2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_169944y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          º
time_distributed_1/ReshapeReshape.conv_lstm2d_1/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:0conv2d_170473conv2d_170475*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_170472 
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_170512conv2d_1_170514*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_170511
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
Ø
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall$^conv_lstm2d/StatefulPartitionedCall&^conv_lstm2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd@ : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2J
#conv_lstm2d/StatefulPartitionedCall#conv_lstm2d/StatefulPartitionedCall2N
%conv_lstm2d_1/StatefulPartitionedCall%conv_lstm2d_1/StatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
Á
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_174287

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"       f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:À
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborReshape:output:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
half_pixel_centers(\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:²
	Reshape_1Reshape;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ
Á
while_cond_172470
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_172470___redundant_placeholder04
0while_while_cond_172470___redundant_placeholder14
0while_while_cond_172470___redundant_placeholder24
0while_while_cond_172470___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
Á
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_174266

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"       f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:À
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborReshape:output:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
half_pixel_centers(\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:²
	Reshape_1Reshape;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°b
Ú
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_170196

inputs7
split_readvariableop_resource: 9
split_1_readvariableop_resource: -
split_2_readvariableop_resource: 
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢while]

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ j
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                u
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
: *
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
: *
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_170070*
condR
while_cond_170069*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          Ê
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ@ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ j
IdentityIdentitytranspose_1:y:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd@ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
þc
Ü
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_174013

inputs7
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢while]

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    t
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                u
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_173887*
condR
while_cond_173886*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          Ê
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd j
IdentityIdentitytranspose_1:y:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
 
_user_specified_nameinputs
¢
±
conv_lstm2d_while_cond_1713734
0conv_lstm2d_while_conv_lstm2d_while_loop_counter:
6conv_lstm2d_while_conv_lstm2d_while_maximum_iterations!
conv_lstm2d_while_placeholder#
conv_lstm2d_while_placeholder_1#
conv_lstm2d_while_placeholder_2#
conv_lstm2d_while_placeholder_34
0conv_lstm2d_while_less_conv_lstm2d_strided_sliceL
Hconv_lstm2d_while_conv_lstm2d_while_cond_171373___redundant_placeholder0L
Hconv_lstm2d_while_conv_lstm2d_while_cond_171373___redundant_placeholder1L
Hconv_lstm2d_while_conv_lstm2d_while_cond_171373___redundant_placeholder2L
Hconv_lstm2d_while_conv_lstm2d_while_cond_171373___redundant_placeholder3
conv_lstm2d_while_identity

conv_lstm2d/while/LessLessconv_lstm2d_while_placeholder0conv_lstm2d_while_less_conv_lstm2d_strided_slice*
T0*
_output_shapes
: c
conv_lstm2d/while/IdentityIdentityconv_lstm2d/while/Less:z:0*
T0
*
_output_shapes
: "A
conv_lstm2d_while_identity#conv_lstm2d/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
:
Ó=

L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_174749

inputs
states_0
states_17
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity

identity_1

identity_2¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
Z
ë
while_body_170070
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0: A
'while_split_1_readvariableop_resource_0: 5
'while_split_2_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource: ?
%while_split_1_readvariableop_resource: 3
%while_split_2_readvariableop_resource: ¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 

e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_174766

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Á
while_cond_173664
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_173664___redundant_placeholder04
0while_while_cond_173664___redundant_placeholder14
0while_while_cond_173664___redundant_placeholder24
0while_while_cond_173664___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
þc
Ü
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_170428

inputs7
split_readvariableop_resource:@9
split_1_readvariableop_resource:@-
split_2_readvariableop_resource:@
identity¢split/ReadVariableOp¢split_1/ReadVariableOp¢split_2/ReadVariableOp¢while]

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    t
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                u
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :z
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype0¾
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:@*
dtype0
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split£
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¡
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_170302*
condR
while_cond_170301*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          Ê
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:dÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd j
IdentityIdentitytranspose_1:y:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿd : : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd 
 
_user_specified_nameinputs
Z
ë
while_body_172691
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
%while_split_readvariableop_resource_0: A
'while_split_1_readvariableop_resource_0: 5
'while_split_2_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
#while_split_readvariableop_resource: ?
%while_split_1_readvariableop_resource: 3
%while_split_2_readvariableop_resource: ¢while/split/ReadVariableOp¢while/split_1/ReadVariableOp¢while/split_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          ®
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ð
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
: *
dtype0Ö
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
: *
dtype0¦
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÅ
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Ç
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¬
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides
¬
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ *
paddingSAME*
strides

	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ |
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ \
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ _
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ ¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒt
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ §

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿ@ :ÿÿÿÿÿÿÿÿÿ@ : : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ :

_output_shapes
: :

_output_shapes
: 
â
È
,__inference_conv_lstm2d_layer_call_fn_172344
inputs_0!
unknown: #
	unknown_0: 
	unknown_1: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_169187
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
"
_user_specified_name
inputs/0
Ñ
Á
while_cond_174108
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_174108___redundant_placeholder04
0while_while_cond_174108___redundant_placeholder14
0while_while_cond_174108___redundant_placeholder24
0while_while_cond_174108___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¨%
±
D__inference_conv2d_1_layer_call_and_return_conditional_losses_170511

inputs?
%conv2d_conv2d_readvariableop_resource:
@
2squeeze_batch_dims_biasadd_readvariableop_resource:

identity¢Conv2D/Conv2D/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@          z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0¸
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ@       
   ¥
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 

)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0·
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"@       
   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:®
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
w
SoftmaxSoftmax%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
l
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 

NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd@ : : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿd@ 
 
_user_specified_nameinputs
Ñ
Á
while_cond_169825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_169825___redundant_placeholder04
0while_while_cond_169825___redundant_placeholder14
0while_while_cond_169825___redundant_placeholder24
0while_while_cond_169825___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ã
serving_default¯
G
input_1<
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿd@ H
conv2d_1<
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿd@ 
tensorflow/serving/predict:¯î
¦
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ã
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
°
	layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
 cell
!
state_spec
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
°
	(layer
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
»

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
»

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer

?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate/m©0mª7m«8m¬Dm­Em®Fm¯Gm°Hm±Im²/v³0v´7vµ8v¶Dv·Ev¸Fv¹GvºHv»Iv¼"
	optimizer
f
D0
E1
F2
G3
H4
I5
/6
07
78
89"
trackable_list_wrapper
f
D0
E1
F2
G3
H4
I5
/6
07
78
89"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
æ2ã
&__inference_model_layer_call_fn_170541
&__inference_model_layer_call_fn_171255
&__inference_model_layer_call_fn_171280
&__inference_model_layer_call_fn_171156À
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
Ò2Ï
A__inference_model_layer_call_and_return_conditional_losses_171793
A__inference_model_layer_call_and_return_conditional_losses_172306
A__inference_model_layer_call_and_return_conditional_losses_171190
A__inference_model_layer_call_and_return_conditional_losses_171224À
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
ÌBÉ
!__inference__wrapped_model_169003input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Oserving_default"
signature_map
è

Dkernel
Erecurrent_kernel
Fbias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T_random_generator
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Wstates
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
,__inference_conv_lstm2d_layer_call_fn_172344
,__inference_conv_lstm2d_layer_call_fn_172355
,__inference_conv_lstm2d_layer_call_fn_172366
,__inference_conv_lstm2d_layer_call_fn_172377Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÿ2ü
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_172597
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_172817
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_173037
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_173257Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¥
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¬2©
1__inference_time_distributed_layer_call_fn_173262
1__inference_time_distributed_layer_call_fn_173267À
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
â2ß
L__inference_time_distributed_layer_call_and_return_conditional_losses_173285
L__inference_time_distributed_layer_call_and_return_conditional_losses_173303À
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
è

Gkernel
Hrecurrent_kernel
Ibias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l_random_generator
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

ostates
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_conv_lstm2d_1_layer_call_fn_173314
.__inference_conv_lstm2d_1_layer_call_fn_173325
.__inference_conv_lstm2d_1_layer_call_fn_173336
.__inference_conv_lstm2d_1_layer_call_fn_173347Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_173569
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_173791
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_174013
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_174235Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¥
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
°2­
3__inference_time_distributed_1_layer_call_fn_174240
3__inference_time_distributed_1_layer_call_fn_174245À
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
æ2ã
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_174266
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_174287À
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
':%2conv2d/kernel
:2conv2d/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_conv2d_layer_call_fn_174296¢
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
ì2é
B__inference_conv2d_layer_call_and_return_conditional_losses_174329¢
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
):'
2conv2d_1/kernel
:
2conv2d_1/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_1_layer_call_fn_174338¢
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
î2ë
D__inference_conv2d_1_layer_call_and_return_conditional_losses_174371¢
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:* 2conv_lstm2d/kernel
6:4 2conv_lstm2d/recurrent_kernel
: 2conv_lstm2d/bias
.:,@2conv_lstm2d_1/kernel
8:6@2conv_lstm2d_1/recurrent_kernel
 :@2conv_lstm2d_1/bias
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
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
$__inference_signature_wrapper_172333input_1"
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
 
5
D0
E1
F2"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¦2£
/__inference_conv_lstm_cell_layer_call_fn_174388
/__inference_conv_lstm_cell_layer_call_fn_174405¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
Ü2Ù
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_174480
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_174555¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_max_pooling2d_layer_call_fn_174560¢
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
ó2ð
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_174565¢
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
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
G0
H1
I2"
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
h	variables
itrainable_variables
jregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
ª2§
1__inference_conv_lstm_cell_1_layer_call_fn_174582
1__inference_conv_lstm_cell_1_layer_call_fn_174599¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
à2Ý
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_174674
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_174749¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
 0"
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_up_sampling2d_layer_call_fn_174754¢
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
ó2ð
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_174766¢
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
 "
trackable_list_wrapper
'
(0"
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
R

 total

¡count
¢	variables
£	keras_api"
_tf_keras_metric
c

¤total

¥count
¦
_fn_kwargs
§	variables
¨	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
 0
¡1"
trackable_list_wrapper
.
¢	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¤0
¥1"
trackable_list_wrapper
.
§	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,
2Adam/conv2d_1/kernel/m
 :
2Adam/conv2d_1/bias/m
1:/ 2Adam/conv_lstm2d/kernel/m
;:9 2#Adam/conv_lstm2d/recurrent_kernel/m
#:! 2Adam/conv_lstm2d/bias/m
3:1@2Adam/conv_lstm2d_1/kernel/m
=:;@2%Adam/conv_lstm2d_1/recurrent_kernel/m
%:#@2Adam/conv_lstm2d_1/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,
2Adam/conv2d_1/kernel/v
 :
2Adam/conv2d_1/bias/v
1:/ 2Adam/conv_lstm2d/kernel/v
;:9 2#Adam/conv_lstm2d/recurrent_kernel/v
#:! 2Adam/conv_lstm2d/bias/v
3:1@2Adam/conv_lstm2d_1/kernel/v
=:;@2%Adam/conv_lstm2d_1/recurrent_kernel/v
%:#@2Adam/conv_lstm2d_1/bias/v±
!__inference__wrapped_model_169003
DEFGHI/078<¢9
2¢/
-*
input_1ÿÿÿÿÿÿÿÿÿd@ 
ª "?ª<
:
conv2d_1.+
conv2d_1ÿÿÿÿÿÿÿÿÿd@ 
¼
D__inference_conv2d_1_layer_call_and_return_conditional_losses_174371t78;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿd@ 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd@ 

 
)__inference_conv2d_1_layer_call_fn_174338g78;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿd@ 
ª "$!ÿÿÿÿÿÿÿÿÿd@ 
º
B__inference_conv2d_layer_call_and_return_conditional_losses_174329t/0;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿd@ 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd@ 
 
'__inference_conv2d_layer_call_fn_174296g/0;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿd@ 
ª "$!ÿÿÿÿÿÿÿÿÿd@ è
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_173569GHIW¢T
M¢J
<9
74
inputs/0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 è
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_173791GHIW¢T
M¢J
<9
74
inputs/0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ï
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_174013GHIG¢D
=¢:
,)
inputsÿÿÿÿÿÿÿÿÿd 

 
p 

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd 
 Ï
I__inference_conv_lstm2d_1_layer_call_and_return_conditional_losses_174235GHIG¢D
=¢:
,)
inputsÿÿÿÿÿÿÿÿÿd 

 
p

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd 
 À
.__inference_conv_lstm2d_1_layer_call_fn_173314GHIW¢T
M¢J
<9
74
inputs/0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ À
.__inference_conv_lstm2d_1_layer_call_fn_173325GHIW¢T
M¢J
<9
74
inputs/0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¦
.__inference_conv_lstm2d_1_layer_call_fn_173336tGHIG¢D
=¢:
,)
inputsÿÿÿÿÿÿÿÿÿd 

 
p 

 
ª "$!ÿÿÿÿÿÿÿÿÿd ¦
.__inference_conv_lstm2d_1_layer_call_fn_173347tGHIG¢D
=¢:
,)
inputsÿÿÿÿÿÿÿÿÿd 

 
p

 
ª "$!ÿÿÿÿÿÿÿÿÿd æ
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_172597DEFW¢T
M¢J
<9
74
inputs/0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 

 
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 æ
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_172817DEFW¢T
M¢J
<9
74
inputs/0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 

 
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 Í
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_173037DEFG¢D
=¢:
,)
inputsÿÿÿÿÿÿÿÿÿd@ 

 
p 

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd@ 
 Í
G__inference_conv_lstm2d_layer_call_and_return_conditional_losses_173257DEFG¢D
=¢:
,)
inputsÿÿÿÿÿÿÿÿÿd@ 

 
p

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd@ 
 ¾
,__inference_conv_lstm2d_layer_call_fn_172344DEFW¢T
M¢J
<9
74
inputs/0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 

 
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ ¾
,__inference_conv_lstm2d_layer_call_fn_172355DEFW¢T
M¢J
<9
74
inputs/0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 

 
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ ¤
,__inference_conv_lstm2d_layer_call_fn_172366tDEFG¢D
=¢:
,)
inputsÿÿÿÿÿÿÿÿÿd@ 

 
p 

 
ª "$!ÿÿÿÿÿÿÿÿÿd@ ¤
,__inference_conv_lstm2d_layer_call_fn_172377tDEFG¢D
=¢:
,)
inputsÿÿÿÿÿÿÿÿÿd@ 

 
p

 
ª "$!ÿÿÿÿÿÿÿÿÿd@ 
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_174674³GHI¢
¢
(%
inputsÿÿÿÿÿÿÿÿÿ 
[¢X
*'
states/0ÿÿÿÿÿÿÿÿÿ 
*'
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "¢
¢~
%"
0/0ÿÿÿÿÿÿÿÿÿ 
UR
'$
0/1/0ÿÿÿÿÿÿÿÿÿ 
'$
0/1/1ÿÿÿÿÿÿÿÿÿ 
 
L__inference_conv_lstm_cell_1_layer_call_and_return_conditional_losses_174749³GHI¢
¢
(%
inputsÿÿÿÿÿÿÿÿÿ 
[¢X
*'
states/0ÿÿÿÿÿÿÿÿÿ 
*'
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "¢
¢~
%"
0/0ÿÿÿÿÿÿÿÿÿ 
UR
'$
0/1/0ÿÿÿÿÿÿÿÿÿ 
'$
0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ö
1__inference_conv_lstm_cell_1_layer_call_fn_174582 GHI¢
¢
(%
inputsÿÿÿÿÿÿÿÿÿ 
[¢X
*'
states/0ÿÿÿÿÿÿÿÿÿ 
*'
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "{¢x
# 
0ÿÿÿÿÿÿÿÿÿ 
QN
%"
1/0ÿÿÿÿÿÿÿÿÿ 
%"
1/1ÿÿÿÿÿÿÿÿÿ Ö
1__inference_conv_lstm_cell_1_layer_call_fn_174599 GHI¢
¢
(%
inputsÿÿÿÿÿÿÿÿÿ 
[¢X
*'
states/0ÿÿÿÿÿÿÿÿÿ 
*'
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "{¢x
# 
0ÿÿÿÿÿÿÿÿÿ 
QN
%"
1/0ÿÿÿÿÿÿÿÿÿ 
%"
1/1ÿÿÿÿÿÿÿÿÿ 
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_174480³DEF¢
¢
(%
inputsÿÿÿÿÿÿÿÿÿ@ 
[¢X
*'
states/0ÿÿÿÿÿÿÿÿÿ@ 
*'
states/1ÿÿÿÿÿÿÿÿÿ@ 
p 
ª "¢
¢~
%"
0/0ÿÿÿÿÿÿÿÿÿ@ 
UR
'$
0/1/0ÿÿÿÿÿÿÿÿÿ@ 
'$
0/1/1ÿÿÿÿÿÿÿÿÿ@ 
 
J__inference_conv_lstm_cell_layer_call_and_return_conditional_losses_174555³DEF¢
¢
(%
inputsÿÿÿÿÿÿÿÿÿ@ 
[¢X
*'
states/0ÿÿÿÿÿÿÿÿÿ@ 
*'
states/1ÿÿÿÿÿÿÿÿÿ@ 
p
ª "¢
¢~
%"
0/0ÿÿÿÿÿÿÿÿÿ@ 
UR
'$
0/1/0ÿÿÿÿÿÿÿÿÿ@ 
'$
0/1/1ÿÿÿÿÿÿÿÿÿ@ 
 Ô
/__inference_conv_lstm_cell_layer_call_fn_174388 DEF¢
¢
(%
inputsÿÿÿÿÿÿÿÿÿ@ 
[¢X
*'
states/0ÿÿÿÿÿÿÿÿÿ@ 
*'
states/1ÿÿÿÿÿÿÿÿÿ@ 
p 
ª "{¢x
# 
0ÿÿÿÿÿÿÿÿÿ@ 
QN
%"
1/0ÿÿÿÿÿÿÿÿÿ@ 
%"
1/1ÿÿÿÿÿÿÿÿÿ@ Ô
/__inference_conv_lstm_cell_layer_call_fn_174405 DEF¢
¢
(%
inputsÿÿÿÿÿÿÿÿÿ@ 
[¢X
*'
states/0ÿÿÿÿÿÿÿÿÿ@ 
*'
states/1ÿÿÿÿÿÿÿÿÿ@ 
p
ª "{¢x
# 
0ÿÿÿÿÿÿÿÿÿ@ 
QN
%"
1/0ÿÿÿÿÿÿÿÿÿ@ 
%"
1/1ÿÿÿÿÿÿÿÿÿ@ ì
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_174565R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_layer_call_fn_174560R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
A__inference_model_layer_call_and_return_conditional_losses_171190
DEFGHI/078D¢A
:¢7
-*
input_1ÿÿÿÿÿÿÿÿÿd@ 
p 

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd@ 

 Ë
A__inference_model_layer_call_and_return_conditional_losses_171224
DEFGHI/078D¢A
:¢7
-*
input_1ÿÿÿÿÿÿÿÿÿd@ 
p

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd@ 

 Ê
A__inference_model_layer_call_and_return_conditional_losses_171793
DEFGHI/078C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿd@ 
p 

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd@ 

 Ê
A__inference_model_layer_call_and_return_conditional_losses_172306
DEFGHI/078C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿd@ 
p

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿd@ 

 ¢
&__inference_model_layer_call_fn_170541x
DEFGHI/078D¢A
:¢7
-*
input_1ÿÿÿÿÿÿÿÿÿd@ 
p 

 
ª "$!ÿÿÿÿÿÿÿÿÿd@ 
¢
&__inference_model_layer_call_fn_171156x
DEFGHI/078D¢A
:¢7
-*
input_1ÿÿÿÿÿÿÿÿÿd@ 
p

 
ª "$!ÿÿÿÿÿÿÿÿÿd@ 
¡
&__inference_model_layer_call_fn_171255w
DEFGHI/078C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿd@ 
p 

 
ª "$!ÿÿÿÿÿÿÿÿÿd@ 
¡
&__inference_model_layer_call_fn_171280w
DEFGHI/078C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿd@ 
p

 
ª "$!ÿÿÿÿÿÿÿÿÿd@ 
¿
$__inference_signature_wrapper_172333
DEFGHI/078G¢D
¢ 
=ª:
8
input_1-*
input_1ÿÿÿÿÿÿÿÿÿd@ "?ª<
:
conv2d_1.+
conv2d_1ÿÿÿÿÿÿÿÿÿd@ 
Ý
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_174266L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 Ý
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_174287L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
 ´
3__inference_time_distributed_1_layer_call_fn_174240}L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ ´
3__inference_time_distributed_1_layer_call_fn_174245}L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ Û
L__inference_time_distributed_layer_call_and_return_conditional_losses_173285L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Û
L__inference_time_distributed_layer_call_and_return_conditional_losses_173303L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ²
1__inference_time_distributed_layer_call_fn_173262}L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ²
1__inference_time_distributed_layer_call_fn_173267}L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ì
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_174766R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_layer_call_fn_174754R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
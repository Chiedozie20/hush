
n_ctx 1500;  n_state 384
ctx =  time-steps

	n_state: 384 frequency bins. 

plan: how much storage is 

$\text{Embedding for position } p =  \Big[  \sin(p f_1), \sin(p f_2), \dots, \sin(p f_{192}), \,  \cos(p f_1), \cos(p f_2), \dots, \cos(p f_{192})  \Big]$
Cost saving measures
- Full LUT
- FP 16
- INT quant
- HW taylor expansion 

384

plan 

p -  position : 1500
i - index : 384
f(i) = f_i

f_i table : 384 x width

compute p\*f_i in FPGA

phase wrap and handle cosines

put into sin lut / taylor

sin lut: 2^width \* width 

e.g 16 









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


icarus
real    0m42.955s
user    0m42.491s
sys     0m1.016s
real    2m33.569s
user    2m43.290s
sys     0m1.282s

verilator 
real    0m55.479s
user    0m55.076s
sys     0m1.373s
real    2m28.426s
user    2m39.119s
sys     0m1.016s

bug :
switch to cosine in 193 instead of 192 due to latency 



first failing position=3
idx | x_value | received | expected | abs_diff
205 |    1273 |     1399 |     1142 |  257
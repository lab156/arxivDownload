# Task: Run the 1806 math.AG files 
This is a task on with the command
```
 time parallel --jobs 90% ./run_latexml.sh ::: ~/test_latexml/*
 ```
## bridges cluster (one processor) 
real    121m14.292s 
user    119m47.113s
sys     1m14.615s    

** bridges cluster (16 processors)
real    22m41.144s (there was a 20 mins timeout)
user    159m40.060s 
sys     1m36.147s     
 
## raspberry pi, raspbian, parallel (done twice results confirmed)
still need to check the .xml produced by both
real    97m37.811s
user    378m38.006s
sys     2m8.805s
### raspberry pi, no tmux fresh install
real    87m52.969s
user    340m3.146s
sys     2m29.362s


## raspberry pi, Arch, parallel
real    113m17.403s                                                             
user    441m0.945s                                                              
sys     2m53.022s     

## Acer Arch , Parallel 
real    174m17.403s                                                             
user    638m27.965s                                                             
sys     4m55.323s    
## Acer Arch, 4 Gb
real    221m33.656s
user    424m2.191s
sys     3m6.353s



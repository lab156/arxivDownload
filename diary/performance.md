# Task: Run the 1806 math.AG files 
This is a task on with the command
{{{
 time parallel --jobs 90% ./run_latexml.sh ::: ~/test_latexml/*
 }}}
 
## raspberry pi, raspbian, parallel
real    97m37.811s
user    378m38.006s
sys     2m8.805s

## raspberry pi, Arch, parallel
real    113m17.403s                                                             
user    441m0.945s                                                              
sys     2m53.022s     

## Acer Arch , Parallel 
real    174m17.403s                                                             
user    638m27.965s                                                             
sys     4m55.323s    


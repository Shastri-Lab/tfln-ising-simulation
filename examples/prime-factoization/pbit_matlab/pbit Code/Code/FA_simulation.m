close all;
clear all;
clc
tic

no_run = 2e4;
beta=5;
E=[];

J = [0 -1 -1 1 2;
    -1 0 -1 1 2;
    -1 -1 0 1 2;
     1 1 1 0 -2;
     2 2 2 -2 0]
h = [0 0 0 0 0];

mi=[];E=[];


mi=[0 0 0 0 0];

for i=1:no_run
    
    [mi,E] = pbit_n(i,J,h,mi,beta,E);
end
Genfig_FA(mi,E);




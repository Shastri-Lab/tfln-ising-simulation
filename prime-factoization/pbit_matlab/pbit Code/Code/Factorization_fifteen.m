close all;
clear all;
clc
tic

no_run = 2e4;
beta=1;
E=[];

h = [58 50 12 -80]
J = [0 25 -6 -64;
     25 0 2  -64;
     -6 2  0   16;
     -64  -64 16 0]
factor = max(abs(h))

h = h/factor
J = J/factor
mi=[];E=[];




mi=zeros(1, length(h));

for i=1:no_run
    
    [mi,E] = pbit_n(i,J,h,mi,beta,E);
end
Genfig_Fac(mi,E);


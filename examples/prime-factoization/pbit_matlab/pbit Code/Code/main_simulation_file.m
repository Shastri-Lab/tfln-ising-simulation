close all;
clear all;
clc
tic

no_run=2e4;
E=0; % store all the energy here
beta=2;
% subfig1_for all sample
% subfig2_for the sample with the minimum energy only
%% Probabilistic Copy gate
%%
J=[0 1;
   1 0];
h=[0 0];
mi=[0 0];
for i=1:no_run
    [mi,E] = pbit_n(i,J,h,mi,beta,E)
end
%% Counting the combination in order to generate probability
%%
Genfig_COPY(mi,E);
% subfig1_for all sample
% subfig2_for the sample with the minimum energy only
%% Probabilistic NOT gate
%%
E=[];mi=[];
J=[0 -1;
   -1 0];
h=[0 0];
mi=[0 0 ];
for i=1:no_run
    [mi,E] = pbit_n(i,J,h,mi,beta,E);
end
%Genfig_NOT(mi,E);
% subfig1_for all sample
% subfig2_for the sample with the minimum energy only
%% Probabilistic AND gate
%%
mi=[];E=[];
J=[ 0 -1 2;
   -1  0 2;
    2  2 0];
h=[1 1 -2];

mi=[0 0 0];
beta=1;E=[];
for i=1:no_run
    
    [mi,E] = pbit_n(i,J,h,mi,beta,E);
end
%Genfig_AND(mi,E);
% subfig1_for all sample
% subfig2_for the sample with the minimum energy only
%% Probabilistic OR gate
%%
J=[ 0 -2 3;
   -2  0 3;
    3  3 0];
h=[-1 -1 2];
mi=[];E=[];
mi=[1 1 1];
beta=1;E=[];
for i=1:no_run
    [mi,E] = pbit_n(i,J,h,mi,beta,E);
    mi;
end
%Genfig_OR(mi,E);
% subfig1_for all sample
% subfig2_for the sample with the minimum energy only
toc

%% Probabilistic Full Adder
%%

J = [ 0 1 -1 1 2;
    -1 0 -1 +1 +2;
    -1 -1 0 +1 +2;
    +1 +1 +1 0 -2;
    -2 +2 + 2 -2 0]
h = [0 0 0 0 0]

mi=[];E=[];
J=[ 0 -1 2;
   -1  0 2;
    2  2 0];
h=[1 1 -2];

mi=[0 0 0 0 0];
beta=1;E=[];
for i=1:no_run
    
    [mi,E] = pbit_n(i,J,h,mi,beta,E);
end
Genfig_FA(mi,E);




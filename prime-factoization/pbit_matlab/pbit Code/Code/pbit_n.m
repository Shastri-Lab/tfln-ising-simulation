function [mi,E] = pbit_n(iter,J,h,mi,beta,E)
%% initialization using the value from prev state
if iter>1
    mi(iter,:)=mi(iter-1,:);
end
for i=1:size(J,1)
    sum_p1=0;
    for j=1:size(J,1)
        sum_p1=sum_p1+J(i,j)*mi(iter,j);%Equation 2_part 1 only
    end
    Ii=sum_p1+h(i); % Equation 2
    rr=2*rand(1)-1; %generate random num between -1 to 1
    mi(iter,i)=sign(tanh(beta*Ii) -rr); % value updates one after one.
end

%% ca
S=0;
for i=1:size(J,1)
    for j=1:size(J,2)
        S=S+0.5*J(i,j)*mi(iter,i)*mi(iter,j);
    end
end
for i=1:length(h)
    S=S+h(i)*mi(iter,i);
end
E(iter)=-S;
end


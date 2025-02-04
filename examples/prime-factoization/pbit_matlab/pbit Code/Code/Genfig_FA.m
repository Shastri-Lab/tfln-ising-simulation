function Genfig_FA(mi,E)
m1_in1=int8(mi(:,1)==1);
m2_in2=int8(mi(:,2)==1);
m3_in3=int8(mi(:,3)==1);
m4_out1=int8(mi(:,4)==1);
m5_out2=int8(mi(:,5)==1);

% Compute the values
val = 2*(2*(2*(2*m1_in1 + m2_in2) + m3_in3) + m4_out1) + m5_out2;

% Count the occurrences of each value from 0 to 31
aa = histc(val, 0:31);

% Generate the labels as decimal values
lbl = categorical(string(0:31));

% Create the bar chart with the labels in ascending order
figure, subplot(211), bar(lbl, aa, 0.4, 'FaceColor', [0 .5 .5])
title("Probabilistic FA Gate")



m1_in1(E~=min(E))=[];
m2_in2(E~=min(E))=[];
m3_in3(E~=min(E))=[];
m4_out1(E~=min(E))=[];
m5_out2(E~=min(E))=[];


val = 2*(2*(2*(2*m1_in1 + m2_in2) + m3_in3) + m4_out1) + m5_out2;
aa=histc(val,0:31);
%lbl = categorical({'00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111','01000', '01001', '01010', '01011', '01100', '01101', '01110', '01111','10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111','11000', '11001', '11010', '11011', '11100', '11101', '11110', '11111'});
lbl = categorical(string(0:31));
subplot(212),bar(lbl,aa,0.4,'FaceColor',[0 .5 .5])
title("Minimum energy state only")
end




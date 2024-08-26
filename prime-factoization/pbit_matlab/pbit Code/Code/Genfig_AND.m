function Genfig_AND(mi,E)
m1_in1=int8(mi(:,1)==1);
m2_in2=int8(mi(:,2)==1);
m3_out1=int8(mi(:,3)==1);

val=2*(2*m1_in1+m2_in2)+m3_out1;
aa=histc(val,0:7)
lbl=categorical({'000','001','010','011','100','101','110','111'});
figure,subplot(211),bar(lbl,aa,0.4,'FaceColor',[0 .5 .5])
title("Probabilistic AND Gate(PAG)")

m1_in1(E~=min(E))=[];
m2_in2(E~=min(E))=[];
m3_out1(E~=min(E))=[];

val=2*(2*m1_in1+m2_in2)+m3_out1;
aa=histc(val,0:7)
lbl=categorical({'000','001','010','011','100','101','110','111'});
subplot(212),bar(lbl,aa,0.4,'FaceColor',[0 .5 .5])
title("Minimum energy state only:PAG")

end




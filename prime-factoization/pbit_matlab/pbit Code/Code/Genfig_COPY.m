function Genfig_COPY(mi,E)
figure,
m1_in1=int8(mi(:,1)==1);
m2_out1=int8(mi(:,2)==1);
val=2*m1_in1+m2_out1;
aa=histc(val,0:3)
lbl=categorical({'00','01','10','11'});
subplot(211),bar(lbl,aa,0.4,'FaceColor',[0 .5 .5])
title("Probabilistic COPY Gate(PCG)")

m1_in1(E~=min(E))=[];
m2_out1(E~=min(E))=[];
val=2*m1_in1+m2_out1;
aa=histc(val,0:3)
lbl=categorical({'00','01','10','11'});
subplot(212),bar(lbl,aa,0.4,'FaceColor',[0 .5 .5])
title("Minimum energy state only:PCG")
end


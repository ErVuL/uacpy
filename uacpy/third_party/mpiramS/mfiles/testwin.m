frq=30:0.1:120;
fc=75;
bw=30;
wind1=sinc((frq-fc)/bw);

plot(frq,wind1,'b')

hold on

wind2=hann((frq-fc),2*bw);

plot(frq,wind2,'r')

plot([30 120],[0 0],'--k')

axis([30 120 -0.399 1.099])


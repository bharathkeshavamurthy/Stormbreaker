

r0=abs(read_complex_binary('radio0/r.out',40e6));
r1=abs(read_complex_binary('radio1/r.out',40e6));
r2=abs(read_complex_binary('radio2/r.out',40e6));
r3=abs(read_complex_binary('radio3/r.out',40e6));
r4=abs(read_complex_binary('radio4/r.out',40e6));

t=30e6:30.25e6;

figure(1);
subplot(3,2,1);
title('radio0')
plot(20*log10(r0(t)))
subplot(3,2,2);
title('radio1')
plot(20*log10(r1(t)))
subplot(3,2,3);
title('radio2')
plot(20*log10(r2(t)))
subplot(3,2,4);
title('radio3')
plot(20*log10(r3(t)))
subplot(3,2,5);
title('radio4')
plot(20*log10(r4(t)))
Ns=128;
Nc=12;
Nps=108;
papr_backoff=0.45;
osr = 20;
%nsampf = 140*(117+7+4) + 10000;
%nsampf = 140*(59+7+4) + 10000;
%nsampf = 140*(45+5+4) + 10000;
nsampf = 140*(36+4+4) + 10000;

a0=sqrt(1/(Ns*Nps));
a1=papr_backoff;
a2=sqrt(1/osr);

if exist('c0') ~= 1

  c0=read_complex_binary('yeahls_chanest_cc0');
  c1=read_complex_binary('yeahls_chanest_cc1');

  tx=read_complex_binary('tx.out');
  rx=read_complex_binary('rx.out');

  sh0=read_complex_binary('sh0.out');
  sh0=sh0*a0*a2/a1;
  sh1=read_complex_binary('sh1.out');
  sh1=sh1*a0*a2/a1;

  %r = read_complex_binary('r.out',c);
  
  Md = read_float_binary('sc_sync0_Md.out');
  %Pd = read_complex_binary('Pd.out',c);
  %Rd = read_complex_binary('sc_sync0_Rd.out',c);
end

c0=reshape(c0,Ns,[]);
c1=reshape(c1,Ns,[]);

%sh0=reshape(sh0,Nps,[]);
%sh1=reshape(sh1,Nps,[]);

Md(Md>2)=2;

if exist('triggerb') == 2
  trv=csvread('triggerb');
  %tr = zeros(length(Md),1);
  %tr(trv)=1.5;
  %trv2=csvread('trigger2')+1;
  %tr2 = zeros(length(Md),1);
  %tr2(trv2)=1.4;
  trv_t=csvread('triggert');
  tr_base=trv-trv_t;
  trv_s=tr_base+csvread('triggers');
  trv_n=tr_base+csvread('triggern');
  %trv_w=tr_base+csvread('triggerw');
  %trv_a=tr_base+csvread('triggera');
end

figure(1)
subplot(2,2,1)
plot(fftshift(20*log10(abs(c0(:,:)))))
subplot(2,2,3);
plot(fftshift(angle(c0(:,:))))
subplot(2,2,2)
plot(fftshift(20*log10(abs(c1(:,:)))))
subplot(2,2,4)
plot(fftshift(angle(c1(:,:))))

figure(2)
rt=(1:nsampf*osr*85);
rr=(1:(nsampf*85));
Mdrr=(1:(nsampf*85+Ns));
subplot(2,1,1)
plot(20*log10(abs(tx(rt))))
subplot(2,1,2)
hold off;
%plot([zeros(Ns,1); 20*log10(abs(rx(rr)))])
plot([[zeros(Ns,1); 20*log10(abs(rx(rr)))] Md(Mdrr)])
%ylim([-100,3])
if exist('trv')
  hold on;
  for i=1:length(trv)
    plot([trv(i) trv(i)]+Ns, [10,-80])
    plot([trv_s(i) trv_s(i)]+Ns, [10,-80])
    plot([trv_n(i) trv_n(i)]+Ns, [10,-80])
  end
end
hold off;

figure(3)
rh=1:length(sh0);
rp=1:length(sh1);
subplot(2,1,1)
scatter(real(sh0(rh)),imag(sh0(rh)))
axis equal;
subplot(2,1,2)
scatter(real(sh1(rp)),imag(sh1(rp)))
axis equal;

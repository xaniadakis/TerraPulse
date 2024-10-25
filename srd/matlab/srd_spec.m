function [F, Pxx, Pyy, L1, L2, R1, R2, gof1, gof2] = srd_spec(t, fs, x, y)

global spectrum_range;
Frange = spectrum_range;

F=[]; Pxx=[]; Pyy=[]; L1=[]; L2=[]; R1=[]; R2=[]; gof1=0; gof2=0;

Pov=50;
Tseg = 20;
modes = 6;

NN=round(fs*Tseg);
if NN>=length(x); NN=length(x); end;
Nfft=NN;
if mod(Nfft,2)~=0; Nfft=Nfft+1; end;
w=hamming(NN);
% w=rectwin(NN);
[Pxx, F]=pwelch(x, w, floor(NN*Pov/100), Nfft, fs);
Pyy=[];
if ~isempty(y)
	[Pyy, F2]=pwelch(y, w, floor(NN*Pov/100), Nfft, fs);
end
ii=find(F>=Frange(1) & F<=Frange(2));

F=F(ii);
[sreq1,sreq2]=get_equalizer(F,t);
Pxx=Pxx(ii);
Pxx=Pxx.*sreq1;
if ~isempty(y)
    Pyy=Pyy(ii);
    Pyy=Pyy.*sreq2;
end

[L1,noiseline,R1,gof1]=sr_fit(F,Pxx,modes);
L2=[];
R2=[];
gof2=0;
if ~isempty(y)
    [L2,noiseline2,R2,gof2]=sr_fit(F,Pyy,modes);
end

end


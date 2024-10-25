function [t,fs,x,y] = get_srd_data(fpath)

x=[];
y=[];

[t,fs,ch,vb,ok] = get_srd_info(fpath);
if ok==0; return; end;
if fs<=0; return; end;

fp=fopen(fpath, 'r');
fseek(fp,512+16,0);
x=double(fread(fp,inf,'uint16'));
fclose(fp);

date1=datenum(2017,8,10,0,0,0);
if t<date1
    MAX_VAL = 65535.0;
elseif t>=date1
    MAX_VAL = 32767.0;
    if find(x(1:10000)>MAX_VAL) % faulty binary shift 1 byte
        fp=fopen(fpath, 'r');
        fseek(fp,512+17,0);
        x=double(fread(fp,inf,'uint16'));
        fclose(fp);
    end
end

N=length(x);
if mod(N,2)~=0; x=x(1:end-1); N=N-1; end;
if ch==0
    x=x*4.096/MAX_VAL-2.048;	% Scale samples (x->Volt)
else
	xx=reshape(x,2,N/2); % assuming always even
	x=(xx(1,:).')*4.096/MAX_VAL-2.048;	% Scale samples (x->Volt)
	y=(xx(2,:).')*4.096/MAX_VAL-2.048;	% Scale samples (x->Volt)
end

% remove dc
x=x-mean(x);
if ch==1; y=y-mean(y); end;

end


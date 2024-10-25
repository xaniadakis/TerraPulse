function [date,fs,ch,vbat,ok] = get_srd_info(fname)

ok=0;
fs=-1;
ch=0;
date=0;
vbat=0;
DATALOGGERID = uint64(hex2dec('CAD0FFE51513FFDC'));

dirinfo=dir(fname);
if dirinfo.bytes<(2*512); return; end;
fp=fopen(fname, 'r');
fseek(fp, 0, 'bof');
ID = uint64(fread(fp, 1, 'uint64'));
if ID~=DATALOGGERID
    print_err(sprintf('File "%s" is not logger record!',fname));
    return;
end

fseek(fp, 8, 'bof');
S=fread(fp, 1, 'uint8');
MN=fread(fp, 1, 'uint8');
H=fread(fp, 1, 'uint8');
DAY=fread(fp, 1, 'uint8');
D=fread(fp, 1, 'uint8');
M=fread(fp, 1, 'uint8');
Y=fread(fp, 1, 'uint8')+1970;
date=datenum(Y,M,D,H,MN,S);
t0 = datenum(2016,1,1,0,0,0);
t1 = datenum(2017,8,1,0,0,0);
t2 = datenum(2018,8,1,0,0,0);

if date>t0 && date<t1 % correct rtc time (KALPAKI NS2)
% time slop days_off per day
% tslop = 840/931;   % seconds-offset per day
tslop = 480/600;   % seconds-offset per day
dt = (date - t0)*(tslop/86400);
date = date - dt;
end

fseek(fp, 15, 'bof');
fs=double(fread(fp, 1, 'float'));
fseek(fp, 19, 'bof');
ch=fread(fp, 1, 'uint8');

fseek(fp, 20, 'bof');
% fseek(fp, 512+4, 'bof');
vbat=fread(fp, 1, 'float');

ok=1;
fclose(fp);

end


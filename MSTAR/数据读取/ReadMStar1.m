clear;
ReadPath = 'D:\第一师范工作\科研\数据\MSTAR\MSTAR-PublicMixedTargets-CD1\MSTAR_PUBLIC_MIXED_TARGETS_CD1\15_DEG\COL2\SCENE1\2S1\';
SavePath = 'C:\Users\loujun\Desktop\新建文件夹\';
FileType = '*.000';


Files = dir([ReadPath FileType]);
NumberOfFiles = length(Files);
my_num=NumberOfFiles;
sar_data=zeros(my_num,158,158);

for k=1:my_num
%FID = fopen(ReadPath,'rb','ieee-be');

 clut_file=Files(k).name;
s2=strcat(ReadPath,clut_file(1,:));


FID = fopen(s2,'rb','ieee-be');
    ImgColumns = 0;
    ImgRows = 0;
    while ~feof(FID)                                % 在PhoenixHeader找到图片尺寸大小
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'NumberOfColumns'))
            ImgColumns = str2double(Text(18:end));
            Text = fgetl(FID);
            ImgRows = str2double(Text(15:end));
            break;
        end
    end
    while ~feof(FID)                                 % 跳过PhoenixHeader
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'[EndofPhoenixHeader]'))
            break
        end
    end
    Mag = fread(FID,ImgColumns*ImgRows,'float32','ieee-be');
    Phase = fread(FID,ImgColumns*ImgRows,'float32','ieee-be');
    Img1 = reshape(Mag,[ImgColumns ImgRows]);
    Img2 = reshape(Phase,[ImgColumns ImgRows]);
    Img=Img1.*exp(1j*Img2);
    
   sar_data(k,:,:)=Img;


%     figure,
%     imagesc( Img);
%         figure,
%     imagesc( Img1);
fclose (FID);

end
 XX=(reshape(sar_data(10,:,:),158,158));
  figure; imagesc(abs(XX));
 yy=fftshift(fft2(XX,512,512));
 figure; imagesc(abs(yy));
 



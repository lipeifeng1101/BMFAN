function [ sd ] = MMSDG( im)
[m n h]=size(im);
if h==3
im=double(im(:,:,2));
end
im=double(im);
[h w]=size(im);
d=zeros(h,w,4);
dd=zeros(h,w,4);

for j=1:1:3
   [Dxx,Dxy,Dyy] = Hessian2D(im,j);
%  Dxx=a;
%    figure,imshow(Dxx,[])
  % [dxx,Dxy,Dyy] = Copy_of_Hessian2D(im,j);
    D1 = imfilter(im,Dxx,'conv');
    
    for i=0:30:150
        ddd=imrotate(Dxx,i,'bilinear');
        di = imfilter(im,ddd,'conv');
        D1=max(D1,di);
    end
    d(:,:,round(j))=D1/(sum(sum(Dxx)));
end

d1= d(:,:,1);
d2= d(:,:,2);
d3= d(:,:,3);
% d4= d(:,:,4);

sd=d1+d2+d3;

sd=(sd-min(min(sd)))/(max(max(sd))-min(min(sd)))*255;
sd=uint8(sd);
sd=histeq(sd);

end


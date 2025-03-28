function [c]=ICGF(img,tpl)
tpl = load('tpl25.mat');
tpl=struct2array(tpl);

% figure,imshow(img,[])
img=double(img);
img1=conv2((img),tpl,'same');
a=img-img1;
a=(a-min(min(a)))/(max(max(a))-min(min(a)))*255;
c=uint8(a);
c=histeq(c);
c=filter2(fspecial('average',3),c);
% figure,imshow(c,[])
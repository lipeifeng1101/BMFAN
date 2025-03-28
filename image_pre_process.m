function [ fea] = image_pre_process( img )
[m, n]=size(img);

% g_x=[-1,0,1;-1,0,1;-1,0,1];
% g_y = [-1,-1,-1;0,0,0;1,1,1];
g_x=[0,-1,0;
    0,0,0;
   0 ,1,0];
g_y = [0,0,0;
    -1,0,1;
    0,0,0];
dx =imfilter(img,g_x,'symmetric');
dy =imfilter(img,g_y,'symmetric');
dx=double(dx);
dy=double(dy);
%g1=(double(dx).^2+double(dy).^2).^0.5;
 %g1=power((power(dx,2)+power(dy,2)),0.5);
 g1=(dx.^2+dy.^2).^(1/2);
prop=3;
temp=max(max(g1))*prop/255;

for i_1=1:m
    for j_1=1:n
        if g1(i_1,j_1)<temp
            dx(i_1,j_1)=0;
            dy(i_1,j_1)=0;
        else
            dx(i_1,j_1)=dx(i_1,j_1)/g1(i_1,j_1);
            dy(i_1,j_1)=dy(i_1,j_1)/g1(i_1,j_1);
        end
    end
end

kernel=fspecial('gaussian',4, 3);
dx=imfilter(dx,kernel,'symmetric');
dy=imfilter(dy,kernel,'symmetric');

gxx=imfilter(dx,g_x,'symmetric');
gxy=imfilter(dx,g_y,'symmetric');
gyx=imfilter(dy,g_x,'symmetric');
gyy=imfilter(dy,g_y,'symmetric');

fea1=gxx+gyy;
fea2=gyx+gxy;
fea=max(fea1,fea2);
end
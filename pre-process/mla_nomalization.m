function hin=mla_nomalization(img,s)
% img=rgb2gray(img);
% % figure,imshow(img,[])contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
% img = (img(113:382,151:600));
save_dir = 'E:\matlab\sift\save\';
% img=(img-min(min(img)))/(max(max(img))-min(min(img)))*255;
% a=double(imresize(img,[100 150]));
tpl=double(ones(s,s));
r=conv2(double(img),(tpl),'same');
% hin=histeq(uint8(r/s/s));
in=0.5*((double(img)-r/s/s)+255);
hin=histeq(uint8(in));
hin=filter2(fspecial('average',3),hin);


% a=double(imresize(img,[100 150]));
% tpl=fspecial('gaussian',[15,15],5);
% r=conv2(double(a),(tpl),'same');
% in=((double(a)-r));
% % in=(in-min(min(in)))/(max(max(in))-min(min(in)))*255;
% 
% hin=histeq(uint8(in));
% hin=filter2(fspecial('average',2),hin);


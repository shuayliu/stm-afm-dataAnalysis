function [ X,Y ] = covertData( data,k,x,y,var)
% 按照x的升序，对数据进行处理后重新排序
%   Detailed explanation goes here
[row,col]=size(data);
Col_A=data(1:row,1);
Col_B=data(1:row,2);
%下面这一句的意思应该是要进行
Col_D=(Col_B-y)/abs(k);
Y=Col_D*var;
X=Col_A-x+Col_D;
%按照x从小到大重新进行排序
[new_X,index]=sort(X,'ascend');
new_Y=Y;
for i=1:1:row
    new_Y(i)=Y(index(i));
end
X=new_X;
Y=new_Y;
end


%%
%延迟时间和时间窗口计算

function [Smean,Sdeltmean,Scor,tau,tw]=CCMethod(data,max_d)
% 本函数用于求延迟时间tau和时间窗口tw
% data：输入时间序列
% max_d：最大时间延迟
% Smean，Sdeltmean,Scor为返回值
% tau：计算得到的延迟时间
% tw：时间窗口
N=length(data);
%时间序列的长度
Smean=zeros(1,max_d);
%初始化矩阵
Scmean=zeros(1,max_d);
Scor=zeros(1,max_d);
sigma=std(data);
%计算序列的标准差
% 计算Smean,Sdeltmean,Scor
for t=1:max_d
    S=zeros(4,4);
    Sdelt=zeros(1,4);
    for m=2:5
        for j=1:4
            r=sigma*j/2;
            Xdt=disjoint(data,t);
            % 将时间序列data分解成t个不相交的时间序列
            s=0;
           for tau=1:t
                N_t=floor(N/t);
                % 分成的子序列长度
                Y=Xdt(:,tau);
                % 每个子序列
                %计算C(1,N/t,r,t),相当于调用Cs1(tau)=correlation_integral1(Y,r)            
                Cs1(tau)=0;
                for ii=1:N_t-1
                    for jj=ii+1:N_t
                        d1=abs(Y(ii)-Y(jj));
                        % 计算状态空间中每两点之间的距离,取无穷范数
                        if r>d1
                            Cs1(tau)=Cs1(tau)+1;            
                        end
                    end
                end
                Cs1(tau)=2*Cs1(tau)/(N_t*(N_t-1));
              
                Z=reconstitution(Y,m,1);
                % 相空间重构
                M=N_t-(m-1); 
                Cs(tau)=correlation_integral(Z,M,r);
                % 计算C(m,N/t,r,t)
                s=s+(Cs(tau)-Cs1(tau)^m);
                % 对t个不相关的时间序列求和
           end            
           S(m-1,j)=s/tau;            
        end
        Sdelt(m-1)=max(S(m-1,:))-min(S(m-1,:));
        % 差量计算
    end
    Smean(t)=mean(mean(S));
    % 计算平均值
    Sdeltmean(t)=mean(Sdelt);
    % 计算平均值
    Scor(t)=abs(Smean(t))+Sdeltmean(t);
end
% 寻找时间延迟tau：即Sdeltmean第一个极小值点对应的t
for i=2:length(Sdeltmean)-1
    if Sdeltmean(i)<Sdeltmean(i-1)&Sdeltmean(i)<Sdeltmean(i+1)
        tau=i;
        break;
    end
end
% 寻找时间窗口tw：即Scor最小值对应的t
for i=1:length(Scor)
    if Scor(i)==min(Scor)
        tw=i;
        break;
    end
end
%%
%时间序列分解
function Data=disjoint(data,t)
% 此函数用于将时间序列分解成t个不相交的时间序列
% data:输入时间序列
% t:延迟，也是不相交时间序列的个数
% Data:返回分解后的t个不相交的时间序列
N=length(data);
%data的长度
for i=1:t
    for j=1:(N/t)
        Data(j,i)=data(i+(j-1)*t);
    end
end
%%
%相空间重构
function Data=reconstitution(data,m,tau)
%该函数用来重构相空间
% m:嵌入空间维数
% tau:时间延迟
% data:输入时间序列
% Data:输出,是m*n维矩阵
%m=tw/tau+1
N=length(data); 
% N为时间序列长度
M=N-(m-1)*tau; 
%相空间中点的个数
Data=zeros(m,M);
for j=1:M
  for i=1:m
  %相空间重构
    Data(i,j)=data((i-1)*tau+j);
  end
end
%关联积分计算
function C_I=correlation_integral(X,M,r)
%该函数用来计算关联积分
%C_I:关联积分的返回值
%X:重构的相空间矢量，是一个m*M的矩阵
%M::M是重构的m维相空间中的总点数
%r:Heaviside 函数中的搜索半径
sum_H=0;
for i=1:M-1
    for j=i+1:M
        d=norm((X(:,i)-X(:,j)),inf);
        %计算相空间中每两点之间的距离，其中NORM(V,inf) = max(abs(V)).
        if r>d    
        %sita=heaviside(r,d);%计算Heaviside 函数之值n
           sum_H=sum_H+1;
        end
    end
end
C_I=2*sum_H/(M*(M-1));%关联积分的值
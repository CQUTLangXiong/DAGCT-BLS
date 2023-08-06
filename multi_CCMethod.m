function [tau_list,m_list]=multi_CCMethod(max_d, type, start_index, end_index)
%本函数用于求多维混沌时间序列的延迟时间和嵌入维数
%数据范围为3001:7000
%if type=='lorenz'
%strcmp字符串匹配，用==会出现报错
if strcmp(type, 'lorenz')
    data=readtable('data/original_data/lorenz.csv')
    data=table2array(data)
    columns_index = [2 4]
    tau_list = zeros(1,3)
    m_list = zeros(1,3)
%elseif type=='rossler'
elseif strcmp(type, 'rossler')
    data=readtable('data/original_data/rossler.csv')
    data=table2array(data)
    columns_index = [2 4]
    tau_list = zeros(1,3)
    m_list = zeros(1,3)
%elseif type=='sea_clutter'
elseif strcmp(type, 'sea_clutter')
    data=readtable('data/original_data/sea_clutter.xlsx')
    data=table2array(data)
    columns_index = [1 14]
    tau_list = zeros(1,14)
    m_list = zeros(1,14)

end
for i=columns_index(1):columns_index(2)
    data_single=data(:,i)
    %取3001:7000
    data_single = data_single(start_index:end_index,:)
    [Smean,Sdeltmean,Scor,tau,tw] = CCMethod(data_single, max_d)
    m = min(floor(tw/tau +1),5)
    if strcmp(type, 'sea_clutter')
        tau_list(i)=tau
        m_list(i)=m
    else
        tau_list(i-1)=tau
        m_list(i-1)=m
    end
end



    

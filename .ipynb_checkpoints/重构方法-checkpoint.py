import numpy as np


def time_delay_embed(x, y , z, t1, m1, t2, m2, t3, m3):
    """
    对时间序列进行时延嵌入

    参数：
    x：时间序列，一维数组
    t：延迟阶数，整数
    m：嵌入维数，整数
    """
    N = len(x)
    list_tm_max = [(m1-1)*t1+1,(m2-1)*t2+1,(m3-1)*t3+1]
    J0 = max(list_tm_max)
    list_Vn = []
    Vn = np.empty((N - J0 + 1, m1+m2+m3))
    while J0<=N:
        list_line = []
        line_x = [x[(J0-m*t1)-1] for m in range(m1)]
        line_y = [y[(J0-m*t2)-1] for m in range(m2)]
        line_z = [z[(J0-m*t3)-1] for m in range(m3)]

        list_line.extend(line_x)
        list_line.extend(line_y)
        list_line.extend(line_z)
        list_Vn.append(list_line)
        J0 += 1
    Vn = np.array(list_Vn).reshape((4, 9))
    return Vn


# 延迟阶数和嵌入维数
t1, t2, t3 = 3, 2, 1
m1, m2, m3 = 3, 3, 3

# 生成形状为(3, 15)的随机数
rand_arr = np.random.randn(10, 3)
rand_arr = (rand_arr * 10).astype(np.int)
print(rand_arr)
x = rand_arr[:,0]
y = rand_arr[:,1]
z = rand_arr[:,2]

Vn = time_delay_embed(x,y,z,t1, m1, t2, m2, t3, m3)
print(Vn)
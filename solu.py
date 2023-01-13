
import scipy.io as scio
import numpy as np 

from matplotlib import pyplot as plt 
numList=[400,800,1200,1600,2000,2400,2800,3200,3600,4000]
bound=[29,37,26,22,21,29,23,19,32,35]
'''
for i in numList:
    data = scio.loadmat(str(i)+"raw.mat")
    data=data['data']
    m=data.shape[0]
    print(m)
    data=data.reshape((m,))
    x=np.arange(0,m)
    plt.plot(x,data)
    plt.show()
    #print(data)
    #break
'''


# 如何分离敲击信号，搞一个1000大小的框，位移到最大的地方，以此为中心
# 两侧套一千
def get_piece(data):
    m=data.shape[0]
    beforeVolume=np.sum(np.abs(data[0:1000]))
    print("beforeVolume  "+str(beforeVolume))
    signal_piece=[]
    decrease=False
    for i in range(100,m,100):
        if(1000+i>m):
            break
        currVolume=np.sum(np.abs(data[i:1000+i]))
        #print("currVolume   "+str(currVolume))
        if(currVolume<beforeVolume and currVolume>800 and decrease==False):
            print("find")
            rightBound=min(m,i+2000)
            leftBound=max(i-1000,0)
            
            if(rightBound-leftBound<3000):
                add_data=np.array([0]*(3000-rightBound+leftBound))
                ans=np.concatenate((data[leftBound:rightBound],add_data))
                signal_piece.append(ans)
            else:
                signal_piece.append(data[leftBound:rightBound])
            x=np.arange(leftBound,rightBound)
            plt.plot(x,data[leftBound:rightBound])
            
            #plt.show()
            decrease=True

            print("currVolume   "+str(currVolume))
            print("leftBound  "+str(leftBound))
            print("rightBound  "+str(rightBound))

        if(currVolume>beforeVolume and currVolume>800):
            decrease=False
        
        beforeVolume=currVolume
    print(len(signal_piece))
    return signal_piece

# 预加重
def pre_fun(x):  # 定义预加重函数
    signal_points=x.shape[0]  # 获取语音信号的长度
    signal_points=int(signal_points)  # 把语音信号的长度转换为整型
    # s=x  # 把采样数组赋值给函数s方便下边计算
    for i in range(1, signal_points, 1):# 对采样数组进行for循环计算
        x[i] = x[i] - 0.98 * x[i - 1]  # 一阶FIR滤波器
    return x  # 返回预加重以后的采样数组

# 分帧
def frame(x, lframe, mframe):  # 定义分帧函数
    signal_length = len(x)  # 获取语音信号的长度
    fn = (signal_length-lframe)/mframe  # 分成fn帧
    fn1 = np.ceil(fn)  # 将帧数向上取整，如果是浮点型则加一
    fn1 = int(fn1)  # 将帧数化为整数
    # 求出添加的0的个数
    numfillzero = (fn1*mframe+lframe)-signal_length
    # 生成填充序列
    fillzeros = np.zeros(numfillzero)
    # 填充以后的信号记作fillsignal
    fillsignal = np.concatenate((x,fillzeros))  # concatenate连接两个维度相同的矩阵
    # 对所有帧的时间点进行抽取，得到fn1*lframe长度的矩阵d
    d = np.tile(np.arange(0, lframe), (fn1, 1)) + np.tile(np.arange(0, fn1*mframe, mframe), (lframe, 1)).T
    # 将d转换为矩阵形式（数据类型为int类型）
    d = np.array(d, dtype=np.int32)
    signal = fillsignal[d]
    
    # signal is a two-dimensional matrix
    #fn1 is the number of frame
    #numfillzero 求出添加的0的个数
    return(signal, fn1, numfillzero)
def add_window(data,lframe,mframe):
    #sr=8000
    #lframe = int(2560)  # 帧长(持续0.025秒)
    #mframe = int(1280)  # 帧移
    # 函数调用，把采样数组、帧长、帧移等参数传递进函数frame，并返回存储于endframe、fn1、numfillzero中
    endframe, fn1, numfillzero = frame(data, lframe, mframe)

    # 对第一帧进行加窗
    hanwindow = np.hanning(lframe)  # 调用汉明窗，把参数帧长传递进去
    hanwindowSum=np.tile(hanwindow,(fn1,1))
    
    signalwindow = endframe*hanwindowSum  # 乘以汉明窗
    print(signalwindow.shape)
    return signalwindow
    #x1 = np.arange(0, lframe, 1)  # 第一帧采样点刻度
    #plt.plot(x1, signalwindow)
    #plt.show()
lframe=256
mframe=128
def process_fft(data,lframe,mframe):
    signal= add_window(data,lframe,mframe)
    
    fft_signal=np.log10(np.abs(np.fft.fft(signal)))
    return fft_signal
def displaySpectrogram(data):
    #plt.title(wavname)
    spect=process_fft(data,lframe,mframe)
    print("spect shape")
    print(spect.shape)
    # spec=[num_frame,fft_res]
    new_spect=spect.T
    fft_len=new_spect.shape[0]
    new_spect=new_spect[:fft_len//2,:]
    print(new_spect.shape)
    #new_spect=[fft_len/2,num_frame]
    return new_spect
    plt.imshow(new_spect, origin="lower", cmap = "jet", aspect = "auto", interpolation = "none")
    #plt.show()
    
#displaySpectrogram(data)

def main():
    for i in numList:

        data = scio.loadmat(str(i)+"raw.mat")
        data=data['data']
        m=data.shape[0]
        print(m)
        data=data.reshape((m,))

        print(np.sum(np.abs(data[52000:53000])))
        print(np.sum(np.abs(data[51000:52000])))
        print(np.sum(np.abs(data[52000:52500])))

        x=np.arange(0,m)
        plt.plot(x,data)
        #plt.show()
        currIndex=i//400-1
        signal_piece=get_piece(data)
        for j in range(0,len(signal_piece)):
            
            ans=displaySpectrogram(signal_piece[j])
            #plt.savefig("./pic/"+str(i)+"raw/pic"+str(j)+'.jpg')
            if(j<=bound[currIndex]):
                np.savetxt("./pic/"+str(i)+"raw/pic"+str(j)+'.txt',ans)
                print("./pic/"+str(i)+"raw/pic"+str(j)+'.txt')
            else:
                np.savetxt("./test/"+str(i)+"raw/pic"+str(j)+'.txt',ans)
                print("./test/"+str(i)+"raw/pic"+str(j)+'.txt')
        
main()



def cceps(x):
    """
    计算复倒谱
    """
    y = np.fft.fft(x)
    return np.fft.ifft(np.log(y))


def icceps(y):
    """
    计算复倒谱的逆变换
    """
    x = np.fft.fft(y)
    return np.fft.ifft(np.exp(x))


def rcceps(x):
    """
    计算实倒谱
    """
    y = np.fft.fft(x)
    return np.fft.ifft(np.log(np.abs(y)))



# 离散余弦变换
def dct(x):
    N = len(x)
    X = np.zeros(N)
    ts = np.array([i for i in range(N)])
    C = np.ones(N)
    C[0] = np.sqrt(2) / 2
    for k in range(N):
        X[k] = np.sqrt(2 / N) * np.sum(C[k] * np.multiply(x, np.cos((2 * ts + 1) * k * np.pi / 2 / N)))
    return X


def idct(X):
    N = len(X)
    x = np.zeros(N)
    ts = np.array([i for i in range(N)])
    C = np.ones(N)
    C[0] = np.sqrt(2) / 2
    for n in range(N):
        x[n] = np.sqrt(2 / N) * np.sum(np.multiply(np.multiply(C[ts], X[ts]), np.cos((2 * n + 1) * np.pi * ts / 2 / N)))
    return x


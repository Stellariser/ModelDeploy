# coding: utf-8
import warnings
import cv2

warnings.filterwarnings("ignore")  # 忽略警告
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image


# 卷积
def convolve(filter, mat, padding, strides):
    """

    :param filter: 卷积核 [k*k]的矩阵
    :param mat: 被padding图像
    :param padding: 上下左右方向的padding值
    :param strides: 步长 [1,1]
    :return:
    """
    result = None
    filter_size = filter.shape  # [3,3]
    mat_size = mat.shape  # [H,W,C]
    if len(filter_size) == 2:
        # TODO 尝试对三维进行卷积
        if len(mat_size) == 3:  # skip -> 转化为灰度图，不对三维进行卷积
            channel = []
            for i in range(mat_size[-1]):  # 每一个channel
                # 图像上方填充padding[0]个0，下方padding[1]个，左侧padding[2]个，右侧padding[3]个
                # 跟与图片加黑框相同
                pad_mat = np.pad(mat[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0, mat_size[0], strides[1]):  # 高
                    temp.append([])
                    for k in range(0, mat_size[1], strides[0]):  # 宽
                        # 卷积核 * padded_mat里对应区域 -> 求和，为一个value
                        val = (filter * pad_mat[j:j + filter_size[0], k:k + filter_size[1]]).sum()
                        temp[-1].append(val)  # 最后一列加入
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:  # TO here
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j:j + filter_size[0], k:k + filter_size[1]]).sum()
                    channel[-1].append(val)

            result = np.array(channel)

    return result


# 对图像隔点取样
def downsample(img, step=2):
    return img[::step, ::step]


# 根据sigma和计算卷积核
def GuassianKernel(sigma, dim):
    '''
    :param sigma: Standard deviation
    :param dim: dimension(must be positive and also an odd number) -> kernel_size
    :return: return the required Gaussian kernel.
    '''
    temp = [t - (dim // 2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2 * sigma * sigma
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + (assistant.T) ** 2) / temp)
    return result


def getDoG(img, n, sigma0, S=None, O=None):
    '''
    :param img: the original img.
    :param sigma0: sigma of the first stack of the first octave. default 1.52 for complicate reasons.
    :param n: how many stacks of feature that you want to extract.
    :param S: how many stacks does every octave have. S must bigger than 3.
    :param k: the ratio of two adjacent stacks' scale.
    :param O: how many octaves do we have.
    :return: the DoG Pyramid
    '''
    # 假设高斯金字塔每组有S = 5层，则高斯差分金字塔就有S-1 = 4，
    # 那我们只能在高斯差分金字塔每组的中间2层图像求极值(边界是没有极值的)，
    # 所以n = 2

    if S is None:
        S = n + 3
    if O is None:
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3

    k = 2 ** (1.0 / n)  # 初始化k
    # 公式
    # (1<<o)意为2的o次方，每一次都向左位移一位，（二进制）每一位代表2的几次方[1,2,4,8...]
    sigma = [[(k ** s) * sigma0 * (1 << o) for s in range(S)] for o in range(O)]  # 得到第O组第S层的sigma[o][s]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]  # 隔[1,2,4,8...]个点取样得到下采样金字塔

    GuassianPyramid = []
    for i in range(O):  # 每一组：金字塔先加一层空
        GuassianPyramid.append([])
        for j in range(S):  # 每一层
            # TODO 更换dim公式
            dim = int(6 * sigma[i][j] + 1)  # dim -> 高斯核大小[奇数] -> padding大小为 dim/2 向下取整
            if dim % 2 == 0:  # 向上取最邻近奇数
                dim += 1
            # TODO 采用分离的高斯卷积
            GuassianPyramid[-1].append(  # 自己写的卷积方法                         //-> 求整除
                convolve(GuassianKernel(sigma[i][j], dim), samplePyramid[i], [dim // 2, dim // 2, dim // 2, dim // 2],
                         [1, 1]))
    # 得到高斯差分金字塔
    DoG = [[GuassianPyramid[o][s + 1] - GuassianPyramid[o][s] for s in range(S - 1)] for o in range(O)]

    return DoG, GuassianPyramid


# 精确调整关键点（泰勒展开）
def adjustLocalExtrema(DoG, o, s, x, y, contrastThreshold, edgeThreshold, sigma, n, SIFT_FIXPT_SCALE):
    SIFT_MAX_INTERP_STEPS = 5  # 最大迭代次数
    SIFT_IMG_BORDER = 5  # 图像边框(padding的宽度)

    point = []

    img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE)
    deriv_scale = img_scale * 0.5
    second_deriv_scale = img_scale
    cross_deriv_scale = img_scale * 0.25

    img = DoG[o][s]
    i = 0
    while i < SIFT_MAX_INTERP_STEPS:  # 迭代次数
        if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
                img.shape[0] - SIFT_IMG_BORDER:
            return None, None, None, None

        img = DoG[o][s]
        prev = DoG[o][s - 1]
        next = DoG[o][s + 1]

        dD = [(img[x, y + 1] - img[x, y - 1]) * deriv_scale,
              (img[x + 1, y] - img[x - 1, y]) * deriv_scale,
              (next[x, y] - prev[x, y]) * deriv_scale]

        v2 = img[x, y] * 2
        dxx = (img[x, y + 1] + img[x, y - 1] - v2) * second_deriv_scale
        dyy = (img[x + 1, y] + img[x - 1, y] - v2) * second_deriv_scale
        dss = (next[x, y] + prev[x, y] - v2) * second_deriv_scale
        dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * cross_deriv_scale
        dxs = (next[x, y + 1] - next[x, y - 1] - prev[x, y + 1] + prev[x, y - 1]) * cross_deriv_scale
        dys = (next[x + 1, y] - next[x - 1, y] - prev[x + 1, y] + prev[x - 1, y]) * cross_deriv_scale

        H = [[dxx, dxy, dxs],
             [dxy, dyy, dys],
             [dxs, dys, dss]]

        X = np.matmul(np.linalg.pinv(np.array(H)), np.array(dD))  # 矩阵乘法

        xi = -X[2]
        xr = -X[1]
        xc = -X[0]

        if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:  # 若三个维度的偏移量都小于0.5，则无视
            break
        # 当它在任一维度上的偏移量大于0.5时(即x或y或 σ)
        # 意味着插值中心已经偏移到它的邻近点上，所以必须改变当前关键点的位置。同时在新的位置上反复插值直到收敛
        y += int(np.round(xc))
        x += int(np.round(xr))
        s += int(np.round(xi))

        i += 1

    if i >= SIFT_MAX_INTERP_STEPS:  # 超过迭代次数
        return None, x, y, s
    # 超过边界
    if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
            img.shape[0] - SIFT_IMG_BORDER:
        return None, None, None, None

    t = (np.array(dD)).dot(np.array([xc, xr, xi]))

    # 舍去对比度低的点
    contr = img[x, y] * img_scale + t * 0.5
    if np.abs(contr) * n < contrastThreshold:
        return None, x, y, s

    # 利用Hessian矩阵的迹和行列式计算主曲率的比值 -> 去除边缘效应
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    if det <= 0 or tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det:
        return None, x, y, s
    # point 包含图像的H，W
    point.append((x + xr) * (1 << o))  # 乘以2^o以得到对应于底层图像的位置
    point.append((y + xc) * (1 << o))
    point.append(o + (s << 8) + (int(np.round((xi + 0.5)) * 255) << 16))
    point.append(sigma * np.power(2.0, (s + xi) / n) * (1 << o) * 2)

    return point, x, y, s


def GetMainDirection(img, r, c, radius, sigma, BinNum):
    """

    :param img: 高斯金字塔中的图片
    :param r: x坐标
    :param c: y坐标
    :param radius: 窗口半径 [int]
    :param sigma: 关键点所在层的sigma值
    :param BinNum: 直方图柱个数
    :return: omax, hist
    """

    expf_scale = -1.0 / (2.0 * sigma * sigma)  # 内个指数

    X = []
    Y = []
    W = []
    temphist = []

    for i in range(BinNum):
        temphist.append(0.0)

    # 图像梯度直方图统计的像素范围
    k = 0  # 所求点的个数
    ###############################################################
    # 这里为了方便计算，H方向为y方向，W方向为x方向，与locateKeypoint方法相反 #
    ###############################################################
    for i in range(-radius, radius + 1):
        y = r + i  # x方向
        if y <= 0 or y >= img.shape[0] - 1:
            continue
        for j in range(-radius, radius + 1):
            x = c + j  # y方向
            if x <= 0 or x >= img.shape[1] - 1:
                continue

            dx = (img[y, x + 1] - img[y, x - 1])  # delta X
            dy = (img[y - 1, x] - img[y + 1, x])  # delta Y

            X.append(dx)
            Y.append(dy)
            W.append((i * i + j * j) * expf_scale)
            k += 1

    length = k

    W = np.exp(np.array(W))  # 求幂
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi  # 求arctan，得到方向
    Mag = (X ** 2 + Y ** 2) ** 0.5  # 两点距离[模长]

    # 计算直方图的每个bin
    for k in range(length):
        bin = int(np.round((BinNum / 360.0) * Ori[k]))  # 得到一个方向柱子，bin为方向
        if bin >= BinNum:
            bin -= BinNum
        if bin < 0:
            bin += BinNum
        temphist[bin] += W[k] * Mag[k]  # 这个方向的长度+=模长

    # 加入高斯平滑[加入权重]
    # 为了防止某个梯度方向角度因受到噪声的干扰而突变，我们还需要对梯度方向直方图进行平滑处理 -> 根据公式
    temp = [temphist[BinNum - 1], temphist[BinNum - 2], temphist[0], temphist[1]]  #
    temphist.insert(0, temp[0])  # 在最前面插入两个值
    temphist.insert(0, temp[1])
    temphist.insert(len(temphist), temp[2])  # 在最后面也插入两个值
    temphist.insert(len(temphist), temp[3])

    hist = []
    # 计算平滑后的梯度，存入hist
    for i in range(BinNum):
        hist.append(
            (temphist[i] + temphist[i + 4]) * (1.0 / 16.0) + (temphist[i + 1] + temphist[i + 3]) * (4.0 / 16.0) +
            temphist[i + 2] * (6.0 / 16.0))

    # 得到主方向
    maxval = max(hist)

    return maxval, hist


# 求解关键点 -> 局部极值检测
def LocateKeyPoint(DoG, sigma, GuassianPyramid, n, BinNum=36, contrastThreshold=0.04, edgeThreshold=10.0):
    """
    # TODO 尝试threshold (Lowe论文中使用0.03，Rob Hess等人实现时使用0.04/S)
    :param DoG: 高斯差分金字塔
    :param sigma: 当前尺度(sigma值)
    :param GuassianPyramid: 高斯金字塔
    :param n: 取特征的层个数
    :param BinNum: 将 0-360度分为36个柱，每个柱包含10度，柱形图共36个柱
    :param contrastThreshold: 对比度阈值，归一化后的图像小于这个值就剔除
    :param edgeThreshold: 边缘阈值，去除边缘特征
    :return:
    """
    SIFT_ORI_SIG_FCTR = 1.52  # 初始sigma
    SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR  # 窗口半径=3sigma
    SIFT_ORI_PEAK_RATIO = 0.8  # 确定主方向阈值，当元素数量多于0.8倍omax的方向会被认定为主方向或辅方向

    SIFT_INT_DESCR_FCTR = 512.0
    # SIFT_FIXPT_SCALE = 48
    SIFT_FIXPT_SCALE = 1  # 一个影响阈值的倍率因子，可以根据图像最大和最小值只差来调整

    KeyPoints = []
    O = len(DoG)  # 组数
    S = len(DoG[0])  # 层数
    for o in range(O):
        for s in range(1, S - 1):  # 跳过最上最下两层
            # 若分层过多(n大)会导致层差值本身变小，所以这里更新阈值
            # 由于输入图像范围是0-255，所以 threshold = 0.5 * contrastThreshold / n * 255 * SIFT_FIXPT_SCALE # TODO 改
            threshold = 0.5 * contrastThreshold / (n * 255 * SIFT_FIXPT_SCALE)
            img_prev = DoG[o][s - 1]  # 上层
            img = DoG[o][s]  # 本层
            img_next = DoG[o][s + 1]  # 下层
            for i in range(img.shape[0]):  # 这里其实i代表H方向，也就是y方向
                for j in range(img.shape[1]):  # 遍历每个像素
                    val = img[i, j]
                    eight_neiborhood_prev = img_prev[max(0, i - 1):min(i + 2, img_prev.shape[0]),
                                            max(0, j - 1):min(j + 2, img_prev.shape[1])]  # 其实取了9个
                    eight_neiborhood = img[max(0, i - 1):min(i + 2, img.shape[0]),
                                       max(0, j - 1):min(j + 2, img.shape[1])]
                    eight_neiborhood_next = img_next[max(0, i - 1):min(i + 2, img_next.shape[0]),
                                            max(0, j - 1):min(j + 2, img_next.shape[1])]
                    # \ -> 续行
                    if np.abs(val) > threshold and \
                            ((val > 0 and (val >= eight_neiborhood_prev).all() and (val >= eight_neiborhood).all() and (
                                    val >= eight_neiborhood_next).all())
                             or (val < 0 and (val <= eight_neiborhood_prev).all() and (
                                            val <= eight_neiborhood).all() and (val <= eight_neiborhood_next).all())):

                        # 精确调整位置
                        point, x, y, layer = adjustLocalExtrema(DoG, o, s, i, j, contrastThreshold, edgeThreshold,
                                                                sigma, n, SIFT_FIXPT_SCALE)
                        if point is None:  # 没找到极值点
                            continue

                        scl_octv = point[-1] * 0.5 / (1 << o)  # 特征点所在组的尺度系数
                        # 确定主方向
                        # 按Lowe的建议，梯度的模值m(x,y)按 σ=1.5σ_oct 的高斯分布加成
                        # 按尺度采样的3σ原则，领域窗口半径为 3x1.5σ_oct。
                        # 主方向，柱形图(平滑后)
                        omax, hist = GetMainDirection(GuassianPyramid[o][layer], x, y,
                                                      int(np.round(SIFT_ORI_RADIUS * scl_octv)),
                                                      SIFT_ORI_SIG_FCTR * scl_octv, BinNum)
                        mag_thr = omax * SIFT_ORI_PEAK_RATIO  # 大于此阈值被认定为主/辅方向
                        for k in range(BinNum):
                            if k > 0:
                                l = k - 1
                            else:
                                l = BinNum - 1
                            if k < BinNum - 1:
                                r2 = k + 1
                            else:
                                r2 = 0
                            if hist[k] > hist[l] and hist[k] > hist[r2] and hist[k] >= mag_thr:
                                bin = k + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[k] + hist[r2])
                                # j是整数，但直方图的极值点一般并不是准确地位于整数位置
                                # 因此这里进行了极值点拟合。对直方图一维二阶泰勒展开，再取导数为0，
                                # 即可拟合出精度更高的极值点，极值点的位置等于 (负的一阶导数/二阶导数)。
                                if bin < 0:
                                    bin = BinNum + bin
                                else:
                                    if bin >= BinNum:
                                        bin = bin - BinNum
                                temp = point[:]
                                temp.append((360.0 / BinNum) * bin)
                                KeyPoints.append(temp)

    return KeyPoints


def calcSIFTDescriptor(img, ptf, ori, scl, d, n, SIFT_DESCR_SCL_FCTR=3.0, SIFT_DESCR_MAG_THR=0.2,
                       SIFT_INT_DESCR_FCTR=512.0, FLT_EPSILON=1.19209290E-07):
    dst = []
    pt = [int(np.round(ptf[0])), int(np.round(ptf[1]))]  # 坐标点取整
    cos_t = np.cos(ori * (np.pi / 180))  # 余弦值
    sin_t = np.sin(ori * (np.pi / 180))  # 正弦值
    bins_per_rad = n / 360.0
    exp_scale = -1.0 / (d * d * 0.5)
    hist_width = SIFT_DESCR_SCL_FCTR * scl
    radius = int(np.round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))
    cos_t /= hist_width
    sin_t /= hist_width

    rows = img.shape[0]
    cols = img.shape[1]

    hist = [0.0] * ((d + 2) * (d + 2) * (n + 2))
    X = []
    Y = []
    RBin = []
    CBin = []
    W = []

    k = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):

            c_rot = j * cos_t - i * sin_t
            r_rot = j * sin_t + i * cos_t
            rbin = r_rot + d // 2 - 0.5
            cbin = c_rot + d // 2 - 0.5
            r = pt[1] + i
            c = pt[0] + j

            if rbin > -1 and rbin < d and cbin > -1 and cbin < d and r > 0 and r < rows - 1 and c > 0 and c < cols - 1:
                dx = (img[r, c + 1] - img[r, c - 1])
                dy = (img[r - 1, c] - img[r + 1, c])
                X.append(dx)
                Y.append(dy)
                RBin.append(rbin)
                CBin.append(cbin)
                W.append((c_rot * c_rot + r_rot * r_rot) * exp_scale)
                k += 1

    length = k
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi
    Mag = (X ** 2 + Y ** 2) ** 0.5
    W = np.exp(np.array(W))

    for k in range(length):
        rbin = RBin[k]
        cbin = CBin[k]
        obin = (Ori[k] - ori) * bins_per_rad
        mag = Mag[k] * W[k]

        r0 = int(rbin)
        c0 = int(cbin)
        o0 = int(obin)
        rbin -= r0
        cbin -= c0
        obin -= o0

        if o0 < 0:
            o0 += n
        if o0 >= n:
            o0 -= n

        # histogram update using tri-linear interpolation
        v_r1 = mag * rbin
        v_r0 = mag - v_r1

        v_rc11 = v_r1 * cbin
        v_rc10 = v_r1 - v_rc11

        v_rc01 = v_r0 * cbin
        v_rc00 = v_r0 - v_rc01

        v_rco111 = v_rc11 * obin
        v_rco110 = v_rc11 - v_rco111

        v_rco101 = v_rc10 * obin
        v_rco100 = v_rc10 - v_rco101

        v_rco011 = v_rc01 * obin
        v_rco010 = v_rc01 - v_rco011

        v_rco001 = v_rc00 * obin
        v_rco000 = v_rc00 - v_rco001

        idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0
        hist[idx] += v_rco000
        hist[idx + 1] += v_rco001
        hist[idx + (n + 2)] += v_rco010
        hist[idx + (n + 3)] += v_rco011
        hist[idx + (d + 2) * (n + 2)] += v_rco100
        hist[idx + (d + 2) * (n + 2) + 1] += v_rco101
        hist[idx + (d + 3) * (n + 2)] += v_rco110
        hist[idx + (d + 3) * (n + 2) + 1] += v_rco111

    # finalize histogram, since the orientation histograms are circular
    for i in range(d):
        for j in range(d):
            idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
            hist[idx] += hist[idx + n]
            hist[idx + 1] += hist[idx + n + 1]
            for k in range(n):
                dst.append(hist[idx + k])

    # copy histogram to the descriptor,
    # apply hysteresis thresholding
    # and scale the result, so that it can be easily converted
    # to byte array
    nrm2 = 0
    length = d * d * n
    for k in range(length):
        nrm2 += dst[k] * dst[k]
    thr = np.sqrt(nrm2) * SIFT_DESCR_MAG_THR

    nrm2 = 0
    for i in range(length):
        val = min(dst[i], thr)
        dst[i] = val
        nrm2 += val * val
    nrm2 = SIFT_INT_DESCR_FCTR / max(np.sqrt(nrm2), FLT_EPSILON)
    for k in range(length):
        dst[k] = min(max(dst[k] * nrm2, 0), 255)

    return dst


# 计算描述符
def calcDescriptors(gpyr, keypoints, SIFT_DESCR_WIDTH=4, SIFT_DESCR_HIST_BINS=8):
    # SIFT_DESCR_WIDTH = 4，描述直方图的宽度
    # SIFT_DESCR_HIST_BINS = 8 8个方向
    d = SIFT_DESCR_WIDTH
    n = SIFT_DESCR_HIST_BINS
    descriptors = []

    for i in range(len(keypoints)):
        kpt = keypoints[i]
        o = kpt[2] & 255
        s = (kpt[2] >> 8) & 255  # 该特征点所在的组序号和层序号
        scale = 1.0 / (1 << o)  # 缩放倍数
        size = kpt[3] * scale  # 该特征点所在组的图像尺寸
        ptf = [kpt[1] * scale, kpt[0] * scale]  # 该特征点在金字塔组中的坐标
        img = gpyr[o][s]  # 该点所在的金字塔图像

        descriptors.append(calcSIFTDescriptor(img, ptf, kpt[-1], size * 0.5, d, n))
    return descriptors


# SIFT算法主方法
def SIFT(img, showDoGimgs=False):
    SIFT_SIGMA = 1.6  # 1.6^2 - 0.5^2 = 1.5^2
    SIFT_INIT_SIGMA = 0.5  # 假设的摄像头的尺度
    sigma0 = np.sqrt(SIFT_SIGMA ** 2 - SIFT_INIT_SIGMA ** 2)  # = 1.52

    n = 3  # 提取 3张图片中的特征 [除去顶与底]

    DoG, GuassianPyramid = getDoG(img, n, sigma0)  # 得到DOG
    if showDoGimgs:  # 展示DOG
        for i in DoG:
            for j in i:
                plt.imshow(j.astype(np.uint8), cmap='gray')
                plt.axis('off')
                plt.show()
    # 关键点定位、计算描述符
    KeyPoints = LocateKeyPoint(DoG, SIFT_SIGMA, GuassianPyramid, n)
    discriptors = calcDescriptors(GuassianPyramid, KeyPoints)

    return KeyPoints, discriptors


def Lines(img, info, color=(255, 0, 0), err=700):
    if len(img.shape) == 2:
        result = np.dstack((img, img, img))  # 在高度方向叠加三张图
    else:
        result = img
    k = 0
    for i in range(result.shape[0]):  # height = 3 * img_height
        for j in range(result.shape[1]):  # weight = weight
            temp = (info[:, 1] - info[:, 0])  # 两图X相减
            A = (j - info[:, 0]) * (info[:, 3] - info[:, 2])  # j-X1 * 高度(Y2-Y1)
            B = (i - info[:, 2]) * (info[:, 1] - info[:, 0])  # i-Y1 * 高度(X2-X1)
            temp[temp == 0] = 1e-9
            t = (j - info[:, 0]) / temp
            e = np.abs(A - B)
            temp = e < err
            if (temp * (t >= 0) * (t <= 1)).any():
                result[i, j] = color
                k += 1
    print(k)

    return result


def drawLines(X1, X2, Y1, Y2, dis, img, num=50):
    info = list(np.dstack((X1, X2, Y1, Y2, dis))[0])  # dstack在高度方向叠加 shape=[点个数, 5]
    info = sorted(info, key=lambda x: x[-1])
    info = np.array(info)
    info = info[:min(num, info.shape[0]), :]
    img = Lines(img, info)

    if len(img.shape) == 2:
        plt.imshow(img.astype(np.uint8), cmap='gray')
    else:
        plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.show()


# 主方法
if __name__ == '__main__':
    origimg = plt.imread('D:\PythonProject\CV\ModelDeploy\Picmatch\ContinusePics/1.jpg')
    # [Height, Weight, Channel]
    if len(origimg.shape) == 3:
        img = origimg.mean(axis=-1)  # 转为灰度图
    else:
        img = origimg
    # 得到关键点与描述符
    keyPoints, discriptors = SIFT(img)
    print(keyPoints[0])
    print(len(keyPoints[0]))
    # keypoints 为nx5 其中n为关键点个数，前两列是H,W值，后面为sigma，特征点方向，梯度幅值
    # discriptors 共128个参数，分别代表 每个关键点的 邻域的 16个格子的 8个方向
    origimg2 = plt.imread('D:\PythonProject\CV\ModelDeploy\Picmatch\ContinusePics/2.jpg')
    if len(origimg.shape) == 3:
        img2 = origimg2.mean(axis=-1)
    else:
        img2 = origimg2
    ScaleRatio = img.shape[0] * 1.0 / img2.shape[0]  # 缩放因子
    # 如果两图大小不一样，根据宽度比例来缩放到一样大小，采用双重采样差值
    img2 = np.array(Image.fromarray(img2).resize((int(round(ScaleRatio * img2.shape[1])), img.shape[0]), Image.BICUBIC))
    keyPoints2, discriptors2 = SIFT(img2, True)

    knn = KNeighborsClassifier(n_neighbors=1)  # 使用K临近，若图1中的一个描述符的位置在图二中周围有1个临近点，则将这两个临近点匹配
    knn.fit(discriptors, [0] * len(discriptors))  # 生成长度为len(descriptors)全是0的数组作为标签，训练一个knn分类器[所有点都为0类]
    match = knn.kneighbors(discriptors2, n_neighbors=1, return_distance=True)  # 匹配，找1个临近点即为一类，把index和distance存入match中
    # match中包含：
    # match[0]返回值distances：第0列元素为与自身的距离(为0)，后面是(n_neighbors - 1)个与之最近的点与参考点的距离 shape=[关键点个数, n_neighbors]
    # match[1]返回值indices：第0列元素为参考点的索引，后面是(n_neighbors - 1)个与之最近的点的索引 shape=[关键点个数, n_neighbors]

    radius1 = np.array(keyPoints)[:, 2]
    radius2 = np.array(keyPoints2)[:, 2]

    mainDirection1 = np.array(keyPoints)[:, 3]
    mainDirection2 = np.array(keyPoints2)[:, 3]

    Distance1 = np.array(keyPoints)[:, 4]
    Distance2 = np.array(keyPoints2)[:, 4]

    keyPoints = np.array(keyPoints)[:, :2]  # 取前两位，分别为x和y坐标
    keyPoints2 = np.array(keyPoints2)[:, :2]
    keyPoints2[:, 1] = img.shape[1] + keyPoints2[:, 1]  # y值=图像1的y值+原关键点y值[长度] 因为要将图像拼接，所以kp2的位置要右移一个img1长度的单位

    # 平衡原图尺寸
    origimg2 = np.array(Image.fromarray(origimg2).resize((img2.shape[1], img2.shape[0]), Image.BICUBIC))
    result = np.hstack((origimg, origimg2))  # 水平方向叠加
    keyPoints = keyPoints[match[1][:, 0]]  # 取到match的index的所有第0列，为原图（图一上关键点的索引）

    # 关键点坐标
    X1 = keyPoints[:, 1]
    X2 = keyPoints2[:, 1]
    Y1 = keyPoints[:, 0]
    Y2 = keyPoints2[:, 0]
    drawLines(X1, X2, Y1, Y2, match[0][:, 0], result)  # 画线

    # Circle
    cv2.circle(result, (X1, Y1), radius1, (255, 0, 255), 1)
    cv2.circle(result, (X2, Y2), radius2, (255, 255, 0), 1)
    cv2.imshow('result', result)

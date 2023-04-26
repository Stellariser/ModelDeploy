import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def jaccard_similarity(set_a, set_b):
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union)

def sorensen_dice_similarity(set_a, set_b):
    intersection = set_a.intersection(set_b)
    return 2 * len(intersection) / (len(set_a) + len(set_b))

def hash_similarity(str_a, str_b):
    return sum(1 for a, b in zip(str_a, str_b) if a == b) / len(str_a)


def calculate_similarity(seq_a, seq_b, method="jaccard"):
    set_a, set_b = set(seq_a), set(seq_b)
    if method == "jaccard":
        return jaccard_similarity(set_a, set_b)
    elif method == "sorensen_dice":
        return sorensen_dice_similarity(set_a, set_b)
    elif method == "hash":
        return hash_similarity(seq_a, seq_b)
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def dynamic_time_warping(seq_a, seq_b):
    len_a, len_b = len(seq_a), len(seq_b)
    dtw_matrix = np.zeros((len_a + 1, len_b + 1))

    for i in range(len_a + 1):
        for j in range(len_b + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = abs(seq_a[i - 1] - seq_b[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    return dtw_matrix[len_a, len_b]

jaccard_result_list = []
sorensen_dice_result_liat = []
hash_result_list = []
dym_list = []

def SequenceMatching(images):

    img1 = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(images[1], cv2.IMREAD_UNCHANGED)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    for i in range(0,len(matches)):


        x1 = kp1[matches[i].queryIdx].pt[0]
        y1 = kp1[matches[i].queryIdx].pt[1]

        x2 = kp2[matches[i].trainIdx].pt[0]
        y2 = kp2[matches[i].trainIdx].pt[1]



        roi1 = img1[int(y1) - 8:int(y1) + 8, int(x1) - 8:int(x1) + 8]
        roi2 = img1[int(y2) - 8:int(y2) + 8, int(x2) - 8:int(x2) + 8]
        sequence1 = roi1.flatten()
        sequence2 = roi2.flatten()

        jaccard_result = calculate_similarity(sequence1, sequence2, method="jaccard")
        sorensen_dice_result = calculate_similarity(sequence1, sequence2, method="sorensen_dice")
        hash_result = calculate_similarity(sequence1, sequence2, method="hash")
        #dym = dynamic_time_warping(sequence1,sequence2)

        jaccard_result_list.append(jaccard_result)
        sorensen_dice_result_liat.append(sorensen_dice_result)
        hash_result_list.append(hash_result)
        #dym_list.append(dym)

    showPlot(jaccard_result_list,sorensen_dice_result_liat,hash_result_list)





def showPlot(Quantlist,Quantlist2,Quantlist3):

    #print(Quantlist)
    # 获取列表的最大长度
    max_length = max(len(Quantlist), len(Quantlist2), len(Quantlist3))
    # 生成序号列表
    index = list(range(1, max_length + 1))
    # 使用 plt.plot 绘制折线图
    plt.plot(index[:len(Quantlist)], Quantlist, marker='o', linestyle='-', linewidth=2, label='Data1')
    plt.plot(index[:len(Quantlist2)], Quantlist2, marker='s', linestyle='--', linewidth=2, label='Data2')
    plt.plot(index[:len(Quantlist3)], Quantlist3, marker='^', linestyle='-.', linewidth=2, label='Data3')
    #plt.plot(index[:len(Quantlist4)], Quantlist4, marker='^', linestyle='-.', linewidth=2, label='Data3')

    # 设置标题和轴标签
    plt.title("Distribution of Numbers in the Lists")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()


    listres1 = find_top_n_indices(Quantlist,50)
    listres2 = find_top_n_indices(Quantlist2, 50)

    intersect = find_intersection(listres1,listres2)
    print(listres1)
    print(listres2)
    print(intersect)


def find_top_n_indices(data, n):
    return sorted(range(len(data)), key=lambda i: data[i], reverse=True)[:n]


def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return set1.intersection(set2)


if __name__ == '__main__':
    imgpath = []
    imgpath2 = []
    img1path = './bmp/1.png'
    img2path = './bmp/2.png'

    img3path = './transformedPic/113.jpg'
    img4path = './transformedPic/143.jpg'


    imgpath.append(img2path)
    imgpath.append(img1path)
    imgpath2.append(img3path)
    imgpath2.append(img4path)

    SequenceMatching(imgpath)

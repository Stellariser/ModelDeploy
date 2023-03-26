import math

def rotate_point(point, angle, center=(1, 1)):

    x, y = point
    cx, cy = center
    angle_rad = math.radians(angle)  # 将角度转换为弧度

    # 将点平移到旋转中心
    x -= cx
    y -= cy

    # 使用旋转矩阵计算旋转后的坐标
    x_new = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_new = x * math.sin(angle_rad) + y * math.cos(angle_rad)

    # 将点平移回原始位置
    x_new += cx
    y_new += cy

    return (x_new, y_new)

if __name__ == '__main__':
    point = (3, 4)
    angle = 30  # 逆时针旋转30度
    rotated_point = rotate_point(point, angle)
    print(rotated_point)
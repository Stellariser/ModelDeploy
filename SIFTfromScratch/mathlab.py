import math

from PIL import Image


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


def make_black_pixels_transparent(png_path, output_path):
    # 打开 PNG 图像
    png_image = Image.open(png_path).convert("RGBA")

    # 创建一个与 PNG 图像尺寸相同的透明图像
    transparent_image = Image.new("RGBA", png_image.size, (0, 0, 0, 0))

    # 遍历 PNG 图像的每个像素
    for y in range(png_image.size[1]):
        for x in range(png_image.size[0]):
            # 获取当前像素的 RGBA 值
            pixel = png_image.getpixel((x, y))

            # 如果像素是纯黑色，设置其透明度为 0（完全透明）
            if pixel == (0, 0, 0, 255):
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), pixel)

    # 保存结果图像
    transparent_image.save(output_path)


def find_trapezoid_vertices(png_path):
    # 打开 PNG 图像
    png_image = Image.open(png_path).convert("RGBA")

    # 图像尺寸
    width, height = png_image.size

    # 初始化顶点坐标
    top_left = None
    top_right = None
    bottom_left = None

    # 自左至右找到最上面一行中第一个非透明像素
    for x in range(width):
        pixel = png_image.getpixel((x, 0))
        if pixel[3] != 0:  # alpha 通道不为 0（不透明）
            top_left = (x, 0)
            break

    # 根据左上角和图像宽度计算右上角
    top_right = (width - top_left[0] - 1, 0)

    # 自左至右找到最下面一行中第一个非透明像素
    for x in range(width):
        pixel = png_image.getpixel((x, height - 1))
        if pixel[3] != 0:  # alpha 通道不为 0（不透明）
            bottom_left = (x, height - 1)
            break

    # 根据左下角和图像宽度计算右下角
    bottom_right = (width - bottom_left[0] - 1, height - 1)

    return top_left, top_right, bottom_left, bottom_right


if __name__ == '__main__':
    # point = (3, 4)
    # angle = 30  # 逆时针旋转30度
    # rotated_point = rotate_point(point, angle)
    # print(rotated_point)
    make_black_pixels_transparent("./transformedPic/1.png", './bmp/1.png')
    make_black_pixels_transparent("./transformedPic/2.png", './bmp/2.png')
    make_black_pixels_transparent("./transformedPic/3.png", './bmp/3.png')
    make_black_pixels_transparent("./transformedPic/4.png", './bmp/4.png')

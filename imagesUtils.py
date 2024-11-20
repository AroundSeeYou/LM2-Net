import os
import random
import shutil

# 指定图片存放的目录和目标目录
source_dir = 'datSets/AppleLeaf9/Scab'
target_dir = 'datSets/AppleLeaf9/test/Scab'

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取所有图片文件的列表
images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# 计算要复制和删除的图片数量
num_to_move = int(len(images) * 0.34)

# 随机选择要复制的图片
images_to_move = random.sample(images, num_to_move)

# 复制选中的图片到目标文件夹并删除原文件
for image in images_to_move:
    # 构建完整的文件路径
    full_image_path = os.path.join(source_dir, image)
    full_target_path = os.path.join(target_dir, image)

    # 复制文件
    shutil.copy2(full_image_path, full_target_path)

    # 删除原文件
    os.remove(full_image_path)
    print(f"Copied and deleted: {image}")

print(f"{len(images_to_move)} images have been copied to '{target_dir}' and deleted from '{source_dir}'.")
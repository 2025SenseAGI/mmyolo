import os
image_dir = "./val"
label_dir = "./val"
# 检查每个图像是否有同名标注文件
img_miss_label_count = 0
for img_file in os.listdir(image_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue  # 跳过非图像文件
    label_file_txt = os.path.splitext(img_file)[0] + ".txt"
    label_file_xml = os.path.splitext(img_file)[0] + ".xml"
    if not os.path.exists(os.path.join(label_dir, label_file_xml)):
    # if not os.path.exists(os.path.join(label_dir, label_file_txt)) and not os.path.exists(os.path.join(label_dir, label_file_xml)):
        print(f"缺失标注：{img_file}")
        os.remove(os.path.join(image_dir, img_file))
        img_miss_label_count += 1

print(f"缺失标注的图像数量：{img_miss_label_count}")

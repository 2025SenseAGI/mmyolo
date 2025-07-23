import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse

def parse_xml(xml_path, class_names):
    """解析单个XML文件，返回图像信息和标注信息"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    image_info = {
        'file_name': root.find('filename').text,
        'height': height,
        'width': width
    }
    
    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in class_names:
            continue
        
        category_id = class_names.index(name) + 1 # COCO category_id从1开始
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 转换为 COCO 的 [xmin, ymin, width, height] 格式
        coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        area = (xmax - xmin) * (ymax - ymin)
        
        annotations.append({
            'bbox': coco_bbox,
            'category_id': category_id,
            'area': area,
            'iscrowd': 0, # 通常设置为0
        })
        
    return image_info, annotations

def get_all_class_names(xml_dir):
    """遍历所有XML文件，提取所有唯一的类别名称"""
    class_names = set()
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"正在从 {xml_dir} 提取类别名称...")
    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            class_names.add(name)
    return sorted(list(class_names))


def convert_voc_to_coco(xml_dir, class_names, output_json_path):
    """主转换函数"""
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 填充 categories
    for i, name in enumerate(class_names):
        coco_dict['categories'].append({
            'id': i + 1,
            'name': name,
            'supercategory': 'none'
        })
    
    image_id_counter = 1
    annotation_id_counter = 1
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"正在转换 {xml_dir} 中的XML文件...")
    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(xml_dir, xml_file)
        image_info, annotations = parse_xml(xml_path, class_names)
        
        # 填充 image 信息
        image_info['id'] = image_id_counter
        coco_dict['images'].append(image_info)
        
        # 填充 annotation 信息
        for ann in annotations:
            ann['id'] = annotation_id_counter
            ann['image_id'] = image_id_counter
            coco_dict['annotations'].append(ann)
            annotation_id_counter += 1
            
        image_id_counter += 1

    print(f"转换完成，正在保存到 {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f, indent=4)
    print("保存成功！")


if __name__ == '__main__':
    # --- 配置您的路径 ---
    # 假设您的数据集根目录是 'my_dataset'
    dataset_root = '.' 
    train_xml_dir = os.path.join(dataset_root, 'train')
    val_xml_dir = os.path.join(dataset_root, 'val')
    
    # 输出的JSON文件路径
    output_dir = os.path.join(dataset_root, 'annotations')
    os.makedirs(output_dir, exist_ok=True)
    
    output_train_json = os.path.join(output_dir, 'instances_train.json')
    output_val_json = os.path.join(output_dir, 'instances_val.json')

    # --- 开始转换 ---
    # 1. 从训练集获取所有类别名称（通常训练集包含所有类别）
    all_classes = get_all_class_names(train_xml_dir)
    print("提取到的所有类别:", all_classes)
    
    # 2. 转换训练集
    convert_voc_to_coco(train_xml_dir, all_classes, output_train_json)
    
    # 3. 转换验证集
    convert_voc_to_coco(val_xml_dir, all_classes, output_val_json)
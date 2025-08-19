import os
import json

# 你的类别顺序
class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
label2id = {name: i for i, name in enumerate(class_names)}

def labelme_to_yolo_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
        json_path = os.path.join(input_dir, filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img_w, img_h = data['imageWidth'], data['imageHeight']
        lines = []
        for shape in data.get('shapes', []):
            label = shape['label']
            if label not in label2id:
                continue
            if shape.get('shape_type', 'rectangle') != 'rectangle':
                continue
            (x1, y1), (x2, y2) = shape['points']
            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
            class_id = label2id[label]
            lines.append(f'{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}')
        txt_name = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

# 用法示例
# labelme_to_yolo_folder('your_labelme_json_folder', 'your_yolo_txt_output_folder')
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <input_folder> <output_folder>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    labelme_to_yolo_folder(input_folder, output_folder)

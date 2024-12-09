import os
import json
import shutil

# 設定路徑
annotations_file = "/home/yptsai/program/object_detection/stevengrove/D3/v3det_2023_v1_train.json"  # 替換為您的 JSON 檔案路徑
dataset_dir = "/storage/SSD-3/yptsai/v3det/"  # 資料集圖片的根目錄
output_dir = "/storage/SSD-3/yptsai/v3det/data/clase"  # 分類後圖片的保存根目錄

# 創建保存分類圖片的根目錄
os.makedirs(output_dir, exist_ok=True)

# 加載標註檔案
with open(annotations_file, "r") as f:
    data = json.load(f)

# 建立類別 ID 到名稱的映射
category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

# 分類圖片
for annotation in data['annotations']:
    category_id = annotation['category_id']
    category_name = category_id_to_name.get(category_id, "")
    
    # 確保類別名稱有效
    if category_name:
        # 找到對應的圖片資訊
        image_info = next((img for img in data['images'] if img['id'] == annotation['image_id']), None)
        if image_info:
            src_path = os.path.join(dataset_dir, image_info['file_name'])
            # 建立子目錄（以類別名稱命名）
            category_dir = os.path.join(output_dir, category_name)
            
            # 確定目標文件路徑
            dst_path = os.path.join(category_dir, os.path.basename(image_info['file_name']))
            
            # 複製圖片到對應的子目錄
            if os.path.exists(src_path):
                os.makedirs(category_dir, exist_ok=True)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)
                print(f"已複製: {src_path} -> {dst_path}")
            else:
                print(f"檔案未找到: {src_path}")

print("圖片分類完成！")

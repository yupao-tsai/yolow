from datasets import load_dataset
from io import BytesIO
import base64
from PIL import Image
import mmengine
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 加载分类树和全局配置
cat_tree_path = '/home/yptsai/program/object_detection/stevengrove/D3/v3det_2023_v1_category_tree.json'
cat_tree = mmengine.load(cat_tree_path)
output_dir = '/storage/SSD-3/yptsai/v3det/data/clases'

# 准备目录
def prepare_directories():
    os.makedirs(output_dir, exist_ok=True)
    for _, cat_name in cat_tree['id2name'].items():
        target_dir = os.path.join(output_dir, cat_name.replace("'", "").replace(",", "_"))
        os.makedirs(target_dir, exist_ok=True)

# 处理单条数据
def process_data_entry(data):
    try:
        # 解码 Base64 图像
        image_data = BytesIO(base64.b64decode(data['image']))
        image = Image.open(image_data)

        # 获取类别名称
        cat_name = cat_tree['id2name'][data['label']]
        target_dir = os.path.join(output_dir, cat_name.replace("'", "").replace(",", "_"))

        # 计算哈希值并保存文件
        image_data.seek(0)
        hash_key = hashlib.sha256(image_data.read()).hexdigest()
        dst_path = os.path.join(target_dir, f"{hash_key}.png")
        image.save(dst_path)
    except Exception as e:
        pass

if __name__ == "__main__":
    # 加载数据集
    dataset = load_dataset("yhcao/V3Det_ImageNet21k_Cls_100")['train']
    
    # 准备目录
    prepare_directories()

    # 进度条
    with tqdm(total=len(dataset), desc="Processing", unit="files") as pbar:
        # 多线程处理
        with ThreadPoolExecutor(max_workers=16) as executor:
            for _ in executor.map(process_data_entry, dataset):
                pbar.update(1)

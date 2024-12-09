import os
import shutil

# 定义源目录和目标目录
dir1 = "/storage/SSD-3/yptsai/v3det/data/clase"  # 替换为实际的 dir1 路径
dir2 = "/storage/SSD-3/yptsai/v3det/data/clases"  # 替换为实际的 dir2 路径

# 遍历 dir1 中的子目录
for subdir in os.listdir(dir1):
    subdir_path = os.path.join(dir1, subdir)
    
    # 检查是否是子目录
    if os.path.isdir(subdir_path):
        # 在 dir2 中找到对应的子目录
        target_subdir_path = os.path.join(dir2, subdir)
        
        # 如果 dir2 中的子目录不存在，创建它
        os.makedirs(target_subdir_path, exist_ok=True)
        
        # 遍历子目录中的文件
        for file_name in os.listdir(subdir_path):
            src_file_path = os.path.join(subdir_path, file_name)
            dst_file_path = os.path.join(target_subdir_path, file_name)
            
            # 移动文件到目标目录
            shutil.move(src_file_path, dst_file_path)
            print(f"Moved: {src_file_path} -> {dst_file_path}")

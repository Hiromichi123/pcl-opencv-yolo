import os

# 批量重命名图片
def batch_rename_images(directory, old_prefix, start_index, end_index, new_prefix, start_number):
    new_directory = os.path.join(directory, new_prefix)
    os.makedirs(new_directory, exist_ok=True)

    for i in range(start_index, end_index + 1):
        old_name = f"{old_prefix}_{i}.jpg"
        old_path = os.path.join(directory, old_name)

        if not os.path.exists(old_path):
            print(f"跳过：文件不存在 {old_name}")
            continue

        new_name = f"{new_prefix}_{start_number}.jpg"
        new_path = os.path.join(new_directory, new_name)

        os.rename(old_path, new_path)
        print(f"移动并重命名：{old_name} -> {new_prefix}/{new_name}")
        start_number += 1

# 批量重命名标签
def batch_rename_labs(directory, old_prefix, start_index, end_index, new_prefix, start_number):
    new_directory = os.path.join(directory, new_prefix)
    os.makedirs(new_directory, exist_ok=True)

    for i in range(start_index, end_index + 1):
        old_name = f"{old_prefix}_{i}.txt"
        old_path = os.path.join(directory, old_name)

        if not os.path.exists(old_path):
            print(f"跳过：文件不存在 {old_name}")
            continue

        new_name = f"{new_prefix}_{start_number}.txt"
        new_path = os.path.join(new_directory, new_name)

        os.rename(old_path, new_path)
        print(f"移动并重命名：{old_name} -> {new_prefix}/{new_name}")
        start_number += 1

if __name__ == "__main__":
    # ==== 手动更改 ====
    old_prefix = "kitchen"                # 原始前缀
    img_directory = "./kitchen"           # 图片所在目录
    lab_directory = "./lab"             # 标签所在目录
    start_index = 481                   # 起始序号
    end_index = 510                      # 结束序号
    start_number = 1                    # 新序号起始值
    new_prefix = "tomato"                 # 图片新前缀
    # ===================================

    batch_rename_images(img_directory, old_prefix, start_index, end_index, new_prefix, start_number)
    batch_rename_labs(lab_directory, old_prefix, start_index, end_index, new_prefix, start_number)

import os
import shutil

LOGS_DIR = "logs"
TARGET_DIR_NAME = "test_smolvla"

for entry in os.listdir(LOGS_DIR):
    time_dir = os.path.join(LOGS_DIR, entry)

    # 只处理 logs 下的目录
    if not os.path.isdir(time_dir):
        continue

    target_path = os.path.join(time_dir, TARGET_DIR_NAME)

    if os.path.isdir(target_path):
        print(f"Removing: {target_path}")
        shutil.rmtree(target_path)
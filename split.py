import os
import shutil
import random
from pathlib import Path

source_folder = 'all_images'
dest_folder = 'final_data'
train_ratio = 0.8  # 80% train, 20% test
image_extensions = ['.jpg', '.jpeg', '.png']

for split in ['train', 'test']:
    for sub in ['images', 'labels']:
        Path(f"{dest_folder}/{split}/{sub}").mkdir(parents=True, exist_ok=True)

all_images = [f for f in os.listdir(source_folder) if Path(f).suffix.lower() in image_extensions]

random.shuffle(all_images)
split_idx = int(len(all_images) * train_ratio)
train_files = all_images[:split_idx]
test_files = all_images[split_idx:]

def copy_files(file_list, split):
    for img_file in file_list:
        base = Path(img_file).stem
        label_file = base + '.txt'
        
        img_src = os.path.join(source_folder, img_file)
        label_src = os.path.join(source_folder, label_file)
        
        img_dst = os.path.join(dest_folder, split, 'images', img_file)
        label_dst = os.path.join(dest_folder, split, 'labels', label_file)

        shutil.copy(img_src, img_dst)

        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"Warning: No label file for {img_file}")

copy_files(train_files, 'train')
copy_files(test_files, 'test')

class_names = ['Y', 'E', 'O', 'F',
          'P', 'Z', 'G', 'Q', 
          'Halo', 'H', 'R', 'NamaAku', 
          'I', 'S', 'J', 'T', 
          'A', 'K', 'U', 'B', 
          'L', 'V', 'C', 'M', 
          'W', 'D', 'N', 'X']
with open(os.path.join(dest_folder, 'data.yaml'), 'w') as f:
    f.write(f"train: {dest_folder}/train/images\n")
    f.write(f"val: {dest_folder}/test/images\n\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {class_names}\n")

print("Done!")
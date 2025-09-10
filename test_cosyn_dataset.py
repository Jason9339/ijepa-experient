#!/usr/bin/env python3

import os
import sys
import datasets
from PIL import Image

def test_cosyn_dataset():
    """測試 CoSyn diagram 資料集載入"""
    
    # 設定路徑
    root_path = "/ceph/work/KLP/zihcilin39/datasets"
    diagram_path = os.path.join(root_path, "CoSyn", "diagram")
    
    print(f"Testing CoSyn diagram dataset from: {diagram_path}")
    
    # 檢查路徑是否存在
    if not os.path.exists(diagram_path):
        print(f"ERROR: Path does not exist: {diagram_path}")
        return False
    
    # 列出資料夾內容
    print(f"Contents of {diagram_path}:")
    for item in os.listdir(diagram_path):
        print(f"  - {item}")
    
    try:
        # 載入資料集
        print("\nLoading dataset...")
        dataset = datasets.load_from_disk(diagram_path)
        print(f"✓ Successfully loaded dataset with {len(dataset)} samples")
        
        # 檢查第一個樣本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nFirst sample keys: {list(sample.keys())}")
            
            # 檢查圖像
            if 'image' in sample:
                img = sample['image']
                print(f"Image type: {type(img)}")
                
                if isinstance(img, Image.Image):
                    print(f"Image size: {img.size}")
                    print(f"Image mode: {img.mode}")
                    print("✓ Image is PIL Image")
                else:
                    print(f"Image data type: {type(img)}")
                    if hasattr(img, 'shape'):
                        print(f"Image shape: {img.shape}")
            
            # 檢查其他欄位
            for key, value in sample.items():
                if key != 'image':
                    print(f"{key}: {type(value)} - {str(value)[:100]}...")
        
        print("\n✓ Dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cosyn_dataset()
    sys.exit(0 if success else 1)

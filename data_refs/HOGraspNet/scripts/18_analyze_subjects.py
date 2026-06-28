import sys, os
sys.path.append(os.getcwd())

import json
import glob
from pathlib import Path
from collections import defaultdict

# HOG_DIR 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HOG_DIR = os.environ.get('HOG_DIR', os.path.join(BASE_DIR, 'refs', 'HOGraspNet'))
LABELING_DATA_DIR = os.path.join(HOG_DIR, 'data', 'labeling_data')

def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def analyze_subjects():
    """HOGraspNet 데이터에서 subject 정보를 분석합니다."""
    
    json_files = glob.glob(os.path.join(LABELING_DATA_DIR, "**", "*.json"), recursive=True)
    
    print(f"Total annotation files: {len(json_files)}")
    print("=" * 80)
    
    subjects = set()
    objects = set()
    grasps = defaultdict(int)
    
    for idx, json_path in enumerate(json_files[:5000], 1):  # 첫 5000개만 샘플링
        try:
            anno = load_json(json_path)
            
            # subject 정보 추출
            if 'capture_info' in anno:
                subject = anno['capture_info'].get('subject_id', anno['capture_info'].get('subject', 'unknown'))
                subjects.add(subject)
            
            # object 정보
            if 'object' in anno:
                obj_id = anno['object'].get('id', 'unknown')
                objects.add(obj_id)
            
            # grasp 정보
            if 'annotations' in anno and len(anno['annotations']) > 0:
                grasp_name = anno['annotations'][0].get('class_name', 'unknown')
                grasps[grasp_name] += 1
            
            if idx % 1000 == 0:
                print(f"Processed {idx} files...")
                
        except Exception as e:
            continue
    
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS (sampled from 5000 files)")
    print("=" * 80)
    
    print(f"\nUnique Subjects: {len(subjects)}")
    for subject in sorted(subjects):
        print(f"  - {subject}")
    
    print(f"\nUnique Objects: {len(objects)}")
    
    print(f"\nGrasp Types: {len(grasps)}")
    for grasp, count in sorted(grasps.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {grasp}: {count}")

if __name__ == '__main__':
    analyze_subjects()

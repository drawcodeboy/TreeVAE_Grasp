# 위험한 코드
import os
import os.path
import sys
from glob import glob
import zipfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time

def unzip_single(zip_file, output_path):
    """Single zip file extraction function for parallel processing"""
    try:
        os.makedirs(output_path, exist_ok=True)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_path)
        return True, f"O {zip_file} -> {output_path}"
    except Exception as e:
        return False, f"X {zip_file} -> ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Unzip HOGraspNet dataset")
    parser.add_argument(
        "--base_path",
        type=str,
        help="Base path containing the zipped files",
        required=False,
        default="data"
    )

    parser.add_argument(
        "--obj_models",
        type=bool,
        default=False,
        help="Whether to unzip object models zip file"
    )

    ## config ##
    args = parser.parse_args()
    base_path = args.base_path

    ## unzip object models if exists
    fname = os.path.join(base_path, "obj_scanned_models", "HOGraspNet_obj_models.zip")
    if os.path.isfile(fname) and args.obj_models:
        print("HOGraspNet_obj_models.zip exists. Unzip.")
        unzip_single(fname, os.path.join(base_path, "obj_scanned_models"))
        # os.remove(fname) # unzip하고 지우는 코드라 우선 안 함 

    ## unzip dataset -> return하는 거 list임
    fnames = glob(os.path.join(base_path, "zipped", "**/*"), recursive=True)
    print("Found the following zip files:")
    zips = [[], [], [], []]

    fname_types = ["Labeling_data","extra_data","Source_augmented","Source_data"]
    fname_outs = ["labeling_data","extra_data","source_augmented","source_data"]

    fname_dict = {
        "Labeling_data": [],
        "extra_data": [],
        "Source_augmented": [],
        "Source_data": []
    }

    # dataset zip 파일들을 유형별로 분류하여 리스트에 저장
    for fname in fnames:
        if not ".zip" in fname:
            continue
        for fname_type, zip_list in fname_dict.items():
            if fname_type in fname:
                zip_list.append(fname)
                break

    '''
    for fname_type, zip_list, fname_out in zip(fname_types, zips, fname_outs):
        output_path = os.path.join(base_path, fname_out)
        # os.makedirs(output_path, exist_ok=True)
 
        pbar = tqdm(zip_list)
        for zip_file in pbar:
            pbar.set_description(f"Unzipping {zip_file} to {output_path}")
            unzip(zip_file, output_path)
            # os.remove(zip_file) # unzip하고 지우는 코드라 우선 안 함
    '''

    for fname_type, zip_list in fname_dict.items():
        # zip list 유형 별로 순회
        fname_out = fname_type.lower()
        output_path = os.path.join(base_path, fname_out)
        
        if not zip_list:
            continue
            
        print(f"\nProcessing {len(zip_list)} {fname_type} files...", flush=True)
        
        # Prepare tasks for parallel processing
        tasks = []
        for zip_file in zip_list:
            # ZIP 파일을 output_path에 직접 풀기 (subject 폴더 없음)
            tasks.append((zip_file, output_path))
        
        # Parallel processing
        num_workers = min(multiprocessing.cpu_count(), len(tasks))
        print(f"Using {num_workers} workers for parallel extraction...", flush=True)
        
        completed_count = 0
        start_time = time.time()
        last_report_time = start_time
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(unzip_single, zip_file, zip_output_path): (zip_file, zip_output_path) 
                            for zip_file, zip_output_path in tasks}
            
            # Progress tracking without tqdm
            for future in as_completed(future_to_task):
                success, result = future.result()
                completed_count += 1
                
                # Print result in main process
                print(result, flush=True)
                
                # Report progress every 10 seconds
                current_time = time.time()
                if current_time - last_report_time >= 10:
                    elapsed = current_time - start_time
                    progress = completed_count / len(tasks) * 100
                    eta = (elapsed / completed_count) * (len(tasks) - completed_count) if completed_count > 0 else 0
                    progress_msg = f"[{fname_type}] {progress:.1f}% complete ({completed_count}/{len(tasks)}) - ETA: {eta:.0f}s"
                    print(progress_msg, flush=True)
                    last_report_time = current_time
        
        # Final completion message
        total_time = time.time() - start_time
        final_msg = f"[{fname_type}] Completed in {total_time:.1f}s"
        print(final_msg, flush=True)


if __name__ == "__main__":
    main()
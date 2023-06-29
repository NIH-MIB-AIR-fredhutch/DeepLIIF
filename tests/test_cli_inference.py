import subprocess
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import cv2
import os
import numpy as np

def test_cli_inference(tmp_path, model_dir_final):
    dir_model = model_dir_final
    dir_input = 'Datasets/Sample_Dataset/test_cli'
    dir_output = tmp_path
    
    fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
    num_input = len(fns_input)
    assert num_input > 0
    
    res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output}',shell=True)
    assert res.returncode == 0
    
    fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
    num_output = len(fns_output)
    assert num_output == num_input * 7


def test_cli_inference_eager(tmp_path, model_dir_final):
    dir_model = model_dir_final
    dir_input = 'Datasets/Sample_Dataset/test_cli'
    dir_output = tmp_path
    
    fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
    num_input = len(fns_input)
    assert num_input > 0
    
    res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --eager-mode',shell=True)
    assert res.returncode == 0
    
    fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
    num_output = len(fns_output)
    assert num_output == num_input * 7


def calculate_ssim(dir_a,dir_b,fns,suffix_a,suffix_b,verbose_freq=50):
    print(suffix_a, suffix_b)
    score = 0
    count = 0
    
    for fn in fns:
        path_a = f"{dir_a}/{fn}{suffix_a}.png"
        assert os.path.exists(path_a), f'path {path_a} does not exist'
        img_a = cv2.imread(path_a)
        # print(img_a.shape)
        
        if suffix_b == 'combine': 
            paths_b = [f"{dir_b}/{fn}fake_BS_{i}.png" for i in range(1,5)]
            imgs_b = []
            for path_b in paths_b:
                assert os.path.exists(path_b), f'path {path_b} does not exist'
                imgs_b.append(cv2.imread(path_b))
            img_b = np.mean(imgs_b,axis=(0))
        else:
            path_b = f"{dir_b}/{fn}{suffix_b}.png"
            assert os.path.exists(path_b), f'path {path_b} does not exist'
            img_b = cv2.imread(path_b)
        # print(img_b.shape)
        
        score += ssim(img_a,img_b, data_range=img_a.max() - img_b.min(), multichannel=True, channel_axis=2)
        count +=1

        if count % verbose_freq == 0 or count == len(fns):
            print(f"{count}/{len(fns)}, running mean SSIM {score / count}")
    return score/count

def test_cli_inference_consistency(tmp_path, model_dir_final):
    dir_model = model_dir_final
    dir_input = 'Datasets/Sample_Dataset/test_cli'
    dirs_output = [tmp_path / 'test1', tmp_path / 'test2']
    
    fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
    num_input = len(fns_input)
    assert num_input > 0
    
    for dir_output in dirs_output:
        res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output}',shell=True)
        assert res.returncode == 0
        
        fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output == num_input * 7
    
    fns = [f for f in os.listdir(dirs_output[0]) if os.path.isfile(os.path.join(dirs_output[0], f)) and f.endswith('png')]
    l_suffix = list(set([fn[:-4].split('_')[-1] for fn in fns]))
    print('suffix:',l_suffix)       

    files = os.listdir(dirs_output[0])
    files = [x for x in files if x.endswith('.png')]
    fns = list(set(['_'.join(x[:-4].split('_')[:-1]) for x in files])) # remove suffix (e.g., fake_B_1.png), then take unique values
    print('num of input files (derived from output):',len(fns))
    print('input img name:',fns)
    
    for i, suffix in enumerate(l_suffix):
        print(f'Calculating {suffix}...')
        ssim_score = calculate_ssim(dirs_output[0], dirs_output[1], fns, '_'+suffix, '_'+suffix)
        print(ssim_score)
        assert ssim_score == 1

def test_cli_inference_eager_consistency(tmp_path, model_dir_final):
    dir_model = model_dir_final
    dir_input = 'Datasets/Sample_Dataset/test_cli'
    dirs_output = [tmp_path / 'test1', tmp_path / 'test2']
    
    fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
    num_input = len(fns_input)
    assert num_input > 0
    
    for dir_output in dirs_output:
        res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --eager-mode',shell=True)
        assert res.returncode == 0
        
        fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output == num_input * 7
    
    fns = [f for f in os.listdir(dirs_output[0]) if os.path.isfile(os.path.join(dirs_output[0], f)) and f.endswith('png')]
    l_suffix = list(set([fn[:-4].split('_')[-1] for fn in fns]))
    print('suffix:',l_suffix)       

    files = os.listdir(dirs_output[0])
    files = [x for x in files if x.endswith('.png')]
    fns = list(set(['_'.join(x[:-4].split('_')[:-1]) for x in files])) # remove suffix (e.g., fake_B_1.png), then take unique values
    print('num of input files (derived from output):',len(fns))
    print('input img name:',fns)
    
    for i, suffix in enumerate(l_suffix):
        print(f'Calculating {suffix}...')
        ssim_score = calculate_ssim(dirs_output[0], dirs_output[1], fns, '_'+suffix, '_'+suffix)
        print(ssim_score)
        assert ssim_score == 1
  

    
    

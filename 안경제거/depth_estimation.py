import os
import glob
import shutil

import numpy as np
import random
import torch
import cv2
import pdb
import torch.nn.functional as F
from math import exp
import torchvision
from PIL import Image
import torchvision.transforms as transforms

def data_generation(all_png = None, seg_png= None, face_png = None, depth_map=None):
    def find_black_boundaries(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        boundaries = max_contour.reshape(-1, 2)
        return boundaries
    def fill_black_area(image):
        boundaries = find_black_boundaries(image)
        fill_image = image.copy()
        cv2.drawContours(fill_image, [boundaries], -1, (255, 255, 255), cv2.FILLED)
        return fill_image
    def remove_white_area(image1, image):
        resized_image2 = cv2.resize(image, (image1.shape[1], image1.shape[0]))
        gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)
        _, binary_image2 = cv2.threshold(gray_image2, 127, 255, cv2.THRESH_BINARY)
        black_image2 = cv2.bitwise_not(binary_image2)
        result = cv2.bitwise_and(image1, image1, mask=black_image2)
        return result
    def find_black_boundaries_all(result):
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    def warp_image(image, x_offset, y_offset, ref_image):
        h, w = ref_image.shape[:2]
        # 원근 변환 행렬 생성
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        # 원근 변환 적용
        warped_image = cv2.warpAffine(image, M, (w, h))
        return warped_image

    if all_png is None:
        image = cv2.imread('/content/drive/MyDrive/image.png')
        face_image = cv2.imread('/content/drive/MyDrive/face.png')
        all_image = image
    else:
        image = seg_png
        face_image = face_png
        all_image = all_png

    fill_image = fill_black_area(image)
    image1 = fill_black_area(fill_image)
    result = remove_white_area(image1, image)
    glass_boundaries = find_black_boundaries_all(result)

    random_factor = random.uniform(2, 3)
    scaling_factor_margin = -0.8           # - 0.1 

    heat_map_margin = random_factor**0.5    # random_factor**0.5
    heat_map_margin2 = random_factor        # random_factor
    background_factor = 2                   # 1
    result_image = None
    result_flow_map_img = None
    
    if len(glass_boundaries) == 2:
        if depth_map is None:
            ############# 이 if문은 depthmap 없을 시절에 쓴 코드다. 그래서 지금은 안쓴다. 
            for i in range(2):
                m = glass_boundaries[i].mean(0)
                new_boundaries = ((glass_boundaries[i] - m) * random_factor + m)
                new_boundaries = new_boundaries.astype(int)
                glass_mask = np.zeros_like(image)
                cv2.drawContours(glass_mask, [new_boundaries], -1, (255, 255, 255), thickness=cv2.FILLED)
                
                result_image_tmp = cv2.bitwise_and(face_image, glass_mask)
                result_image_tmp = cv2.resize(result_image_tmp, None, fx=1/(random_factor+scaling_factor_margin), fy=1/(random_factor+scaling_factor_margin))

                result_image_tmp = warp_image(result_image_tmp, m[0][0]*(1-1/(random_factor+scaling_factor_margin)),m[0][1]*(1-1/(random_factor+scaling_factor_margin)), face_image)
                
                if result_image is None:
                    result_image = result_image_tmp
                else:
                    result_image = cv2.add(result_image, result_image_tmp)
                
            mask_all = result + image
            result_image = cv2.bitwise_and(result_image, result)
            
            reverse_mask_all = 255 - mask_all
            # face_image에서 reverse_mask_all의 영역만 남김
            background_image = cv2.bitwise_and(face_image, reverse_mask_all)
            # to do
            # image는 안경 마스크 (~~seg.png)
            glass_image = cv2.bitwise_and(all_image, image)
            
            composite_image = cv2.add(result_image, glass_image)
            composite_image = cv2.add(composite_image, background_image)
            return composite_image


        else:
            for i in range(2):
                m = glass_boundaries[i].mean(0)

                glass_mask_ori = np.zeros_like(image)
                cv2.drawContours(glass_mask_ori, [glass_boundaries[i]], -1, (255, 255, 255), thickness=cv2.FILLED)

                new_boundaries = ((glass_boundaries[i] - m) * random_factor + m)
                new_boundaries = new_boundaries.astype(int)
                glass_mask = np.zeros_like(image)
                cv2.drawContours(glass_mask, [new_boundaries], -1, (255, 255, 255), thickness=cv2.FILLED)
                glass_mask_2d = glass_mask[:,:,0]


                #'''
                depth_map_lens_min = depth_map[glass_mask_ori[:,:,0]==255].min()
                depth_map_lens_max = depth_map[glass_mask_ori[:,:,0]==255].max()

                glass_max = cv2.max(glass_boundaries[0].max(0), glass_boundaries[1].max(0))[0]
                glass_min = cv2.min(glass_boundaries[0].min(0), glass_boundaries[1].min(0))[0]
                depth_map_nose = depth_map[glass_min[0]:glass_max[0],glass_min[1]:glass_max[1]].min()
                depth_map_eye = depth_map[int(m[0][0]),int(m[0][1])]
                depth_map_gap = depth_map_eye - depth_map_nose

                # closer object has a higher value 
                if depth_map_gap < 0 :
                    depth_map_nose = depth_map[glass_min[0]:glass_max[0],glass_min[1]:glass_max[1]].max()
                    depth_map_gap = depth_map_nose - depth_map_eye
                    depth_map = depth_map.max() - depth_map
                

                # depth_map_scaled = -(depth_map - depth_map_eye - depth_map_gap/4)/depth_map_gap * 10
                #'''

                # basic_flow = - (glass_boundaries[i] - m) * (random_factor-1)
                # depth_aware_scaling_factor = 1 + (random_factor-1)* torch.nn.functional.sigmoid(depth_map_scaled)
                # depth_aware_flow = - (glass_boundaries[i] - m) * (depth_aware_scaling_factor-1)
                # short form

                # depth_aware_scaling_factor = torch.nn.functional.sigmoid(torch.from_numpy(depth_map_scaled))
                x_space = torch.linspace(0,face_image.shape[0]-1,face_image.shape[0])
                y_space = torch.linspace(0,face_image.shape[1]-1,face_image.shape[1])
                meshx, meshy = torch.meshgrid((x_space, y_space))

                delta_flow_map = torch.stack((meshy-m[0][0], meshx-m[0][1]), 2)
                delta_flow_map = delta_flow_map*(random_factor-1)*torch.from_numpy(glass_mask_2d==255).unsqueeze(2)
                depth_aware_flow = delta_flow_map
                # depth_aware_flow = (delta_flow_map) * (depth_aware_scaling_factor.unsqueeze(2))

                # normalized for -1 to 1
                depth_aware_flow[:,:,0] = depth_aware_flow[:,:,0]/face_image.shape[0]/2
                depth_aware_flow[:,:,1] = depth_aware_flow[:,:,1]/face_image.shape[1]/2

                depth_aware_flow = depth_aware_flow.numpy()
                depth_aware_flow = cv2.resize(depth_aware_flow, None, fx=1/(random_factor+scaling_factor_margin), fy=1/(random_factor+scaling_factor_margin))
                depth_aware_flow = warp_image(depth_aware_flow, m[0][0]*(1-1/(random_factor+scaling_factor_margin)),m[0][1]*(1-1/(random_factor+scaling_factor_margin)), face_image)
                depth_aware_flow = torch.from_numpy(depth_aware_flow)
                
                heatmap_scaling_factor = cv2.resize(glass_mask_ori, None, fx=1/(heat_map_margin+scaling_factor_margin), fy=1/(heat_map_margin+scaling_factor_margin))
                heatmap_scaling_factor = warp_image(heatmap_scaling_factor, m[0][0]*(1-1/(heat_map_margin+scaling_factor_margin)),m[0][1]*(1-1/(heat_map_margin+scaling_factor_margin)), face_image)
                heatmap_scaling_factor = torch.from_numpy(heatmap_scaling_factor.transpose(2,0,1)).unsqueeze(0)
                heatmap_scaling_factor = torchvision.transforms.functional.gaussian_blur(heatmap_scaling_factor, kernel_size = (41,41), sigma= (15,15))
                heatmap_scaling_factor = heatmap_scaling_factor.squeeze().permute(1,2,0)[:,:,0]/heatmap_scaling_factor.max()

                depth_aware_scaling_factor = 1 - (depth_map > depth_map_lens_max + depth_map_gap/4).astype(np.float32)
                depth_aware_scaling_factor = cv2.resize(depth_aware_scaling_factor, None, fx=1/(heat_map_margin2+scaling_factor_margin), fy=1/(heat_map_margin2+scaling_factor_margin))
                depth_aware_scaling_factor = warp_image(depth_aware_scaling_factor, m[0][0]*(1-1/(heat_map_margin2+scaling_factor_margin)),m[0][1]*(1-1/(heat_map_margin2+scaling_factor_margin)), face_image)
                depth_aware_scaling_factor = torch.from_numpy(depth_aware_scaling_factor).unsqueeze(0).unsqueeze(0)
                # depth_aware_scaling_factor = torch.from_numpy(depth_aware_scaling_factor.transpose(2,0,1)).unsqueeze(0)
                depth_aware_scaling_factor = torchvision.transforms.functional.gaussian_blur(depth_aware_scaling_factor, kernel_size = (21,21), sigma= (3,3))
                depth_aware_scaling_factor = depth_aware_scaling_factor.squeeze()

                heatmap_scaling_factor = heatmap_scaling_factor*depth_aware_scaling_factor  + background_factor*(1-depth_aware_scaling_factor)
                depth_aware_flow = depth_aware_flow*heatmap_scaling_factor.unsqueeze(2)



                x_space = torch.linspace(-1,1,face_image.shape[0])
                y_space = torch.linspace(-1,1,face_image.shape[1])
                meshx, meshy = torch.meshgrid((x_space, y_space))
                flow_map = torch.stack((meshy, meshx), 2)
                flow_map = flow_map + depth_aware_flow
                
                flow_map_img = torch.cat((depth_aware_flow, torch.zeros_like(depth_aware_flow[:,:,0:1])),2)
                flow_map_img = (flow_map_img.numpy() * 256 )*(glass_mask_ori==255) # Note: 256 means h or w of face_image


                if result_flow_map_img is None:
                    result_flow_map_img = flow_map_img
                else:
                    result_flow_map_img = result_flow_map_img + flow_map_img + 127
                    result_flow_map_img = np.clip(result_flow_map_img, 0, 255).astype(np.uint8)
                    result_flow_map_img = Image.fromarray(result_flow_map_img)

                # flow_map_img를 텐서로 변환하는 코드 추가
                to_tensor = transforms.ToTensor()
                flow_map_tensor = to_tensor(result_flow_map_img)
                
                flow_map = flow_map.unsqueeze(0)
                result_image_tmp = F.grid_sample(torch.from_numpy((face_image).transpose(2,0,1)).float().unsqueeze(0), flow_map.float(), mode='bilinear', align_corners=True)
                result_image_tmp = result_image_tmp.squeeze(0).numpy().transpose(1,2,0)
                result_image_tmp = result_image_tmp*(glass_mask_ori==255)

                if result_image is None:
                    result_image = result_image_tmp
                else:
                    result_image = cv2.add(result_image, result_image_tmp)

            
            # cv2_imshow(result_image)
            mask_all = result + image
            result_image = cv2.bitwise_and(result_image.astype(np.uint8), result)
            
            reverse_mask_all = 255 - mask_all
            # face_image에서 reverse_mask_all의 영역만 남김
            background_image = cv2.bitwise_and(face_image, reverse_mask_all)
            # to do
            # image는 안경 마스크 (~~seg.png)
            glass_image = cv2.bitwise_and(all_image, image)
            # 합성 수행
            composite_image = cv2.add(result_image, glass_image)
            composite_image = cv2.add(composite_image, background_image)
            #cv2_imshow(composite_image)
            return composite_image, result_flow_map_img
    else:
        return None, None


# 파일이 있는 디렉토리 경로
directory = './ALIGN_RESULT_v2'

# 디렉토리 내의 모든 파일 가져오기
files = glob.glob(os.path.join(directory, '*'))



filtered_files = []

end_formats = ['-all.png', '-face.png', '-glass.png', '-seg.png', '-shadow.png']

file_data = [] #4 파일들을 담을 2차원 리스트

for file in files:
    filename = os.path.basename(file)
    _, extension = os.path.splitext(filename) # 파일명을 공백으로 분리하여 단어리스트 생성
    

    if filename.endswith('-all.png') and len(extension) > 0:
        filtered_files.append(file)

def get_filename_from_path(file_paths):
    file_names = []
    for path in file_paths:
        file_name = os.path.basename(path)
        file_names.append(file_name)
    return file_names

shortened_names =  [file[:-8] for file in filtered_files]

shortened_names = get_filename_from_path(shortened_names)

# 새로운 파일명 리스트를 순회하면서 파일들을 찾아서 2차원 리스트에 
for name in shortened_names:
    file_path = os.path.join(directory, name + '.png')
    if os.path.exists(file_path):
        os.remove(file_path)
    folder_path = os.path.join(directory,name + '-')
    if os.path.exists(folder_path):
        # for end_format in end_formats:
        #     file_path = os.path.join(directory, name + end_format)
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        shutil.rmtree(folder_path)


#5 폴더 생성 
input_directory = './input_shadow'
input_original_all_png_directory = './input_original_all_png'
target_directory = './target'
'''
if not os.path.exists(target_directory):
    # shutil.rmtree(folder_path)
    os.makedirs(target_directory)

if not os.path.exists(input_directory):
    # shutil.rmtree(folder_path)
    os.makedirs(input_directory)

if not os.path.exists(input_original_all_png_directory):
    # shutil.rmtree(folder_path)
    os.makedirs(input_original_all_png_directory)

for i, files in enumerate(shortened_names):
    folder_name = shortened_names[i]
    folder_path = os.path.join(target_directory, folder_name)
    
    # 파일들을 폴더에 복사
    # for file in files:
    #    shutil.copy2(file, folder_path)
'''
#6 '.png'> target
source_directory = "./ALIGN_RESULT_v2" # 원본 파일이 있는 경로

# 파일을 복사하고 이름을 변경하는 코드
i = 0

repo = "isl-org/ZoeDepth"
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
zoe = model_zoe_n.to('cuda')


#수정
output_folder = './depth_map/'  # 이미지 파일을 저장할 폴더 경로

for name in shortened_names:
    source_file  = os.path.join(source_directory, name + '-face.png')
    destination_file = os.path.join(target_directory, name + '.png')
    # print(destination_file)
    all_png = cv2.imread(os.path.join(source_directory, name + '-all.png'))
    shadow_png = cv2.imread(os.path.join(source_directory, name + '-shadow.png'))
    seg_png = cv2.imread(os.path.join(source_directory, name + '-seg.png'))
    face_png = cv2.imread(os.path.join(source_directory, name + '-face.png'))

    gen_png = None


    depth_numpy = zoe.infer_pil(face_png)  # as numpy

    #수정 
    file_name = f"depth_image_{name}.png"

    # print(all_png.shape,seg_png.shape,shadow_png.shape,depth_numpy.shape)
    # gen_png, flow_map_img = data_generation(all_png,seg_png,shadow_png,depth_numpy)
    # gen_png = data_generation(all_png,seg_png,shadow_png,depth_numpy)

    try:
        # gen_png = data_generation(all_png,seg_png,shadow_png)
        # gen_png = data_generation(all_png,seg_png,shadow_png,depth_numpy)
        gen_png, flow_map_img = data_generation(all_png,seg_png,shadow_png,depth_numpy)

        if gen_png is not None: cv2.imwrite(os.path.join('depth_aware_input_test', name + '.png'),gen_png)
        flow_map_img.save(os.path.join('./flowMap', name + '.png'))
        # shutil.copy(os.path.join(source_directory, name + '-seg.png'), os.path.join('./gMask', name + '.png'))
        # shutil.copy(os.path.join(source_directory, name + '-glass.png'), os.path.join('./deshadow', name + '.png'))

        #수정
        # output = torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='bilinear', align_corners=True)



        # depth map
        # depth_numpy = depth_numpy * 100
        # print(depth_numpy)
       
        #수정 
        # output_file = os.path.join(output_folder, file_name)
        # cv2.imwrite(output_file, depth_numpy)
        
        
    except:
        i +=1

        #수정 
        #imshow(face_png)
        

        print("error: ", name)


'''
print("Files saved successfully.")

#4 결과 출력
for i, files in enumerate(file_data):
    print(f"Files for name {shortened_names[i]}:")
    for file in files:
        print(file)
    print()
#'''

#3 정제된 파일명 출력
#for name in shortened_names:
#    print(name)

print(f"조건에 맞는 파일 수 : {len(filtered_files)}, 저장된 파일 수:", len(filtered_files)-i)

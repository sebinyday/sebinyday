def rotate_image_90_degrees(image):
    N = len(image)
    # 전치
    for i in range(N):
        for j in range(i, N):
            image[i][j], image[j][i] = image[j][i], image[i][j]
    
    # 각 행 뒤집기
    for i in range(N):
        image[i].reverse()

    return image

#원본 
image = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

#전치 후
image = [
    [1, 4, 7],
    [2, 5, 8],
    [3, 6, 9]
]

#각 행 뒤집기
image = [
    [7, 4, 1],
    [8, 5, 2],
    [9, 6, 3]
]



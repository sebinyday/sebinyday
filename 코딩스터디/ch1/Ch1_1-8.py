def set_zeros(matrix):
    rows, cols = len(matrix), len(matrix[0])
    zero_rows, zero_cols = set(), set() #0이 있는 행과열의 인덱스 저장하는 집합

    # 0의 위치 찾기
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                zero_rows.add(i)    #0이 있으면 각 집합에 추가
                zero_cols.add(j)

    # 해당 행과 열을 0으로 설정
    for i in zero_rows:
        for j in range(cols):
            matrix[i][j] = 0

    for j in zero_cols:
        for i in range(rows):
            matrix[i][j] = 0

    return matrix
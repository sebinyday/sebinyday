def set_zeros(matrix):
    rows, cols = len(matrix), len(matrix[0])
    zero_rows, zero_cols = set(), set() #0�� �ִ� ������� �ε��� �����ϴ� ����

    # 0�� ��ġ ã��
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                zero_rows.add(i)    #0�� ������ �� ���տ� �߰�
                zero_cols.add(j)

    # �ش� ��� ���� 0���� ����
    for i in zero_rows:
        for j in range(cols):
            matrix[i][j] = 0

    for j in zero_cols:
        for i in range(rows):
            matrix[i][j] = 0

    return matrix
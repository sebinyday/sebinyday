# ���ڿ��� �ΰ� �־����� ��, ���� ���� ���迡 �ִ��� Ȯ���ϴ� �޼��� 

# �� Ǯ�� 

def is_permutation(s1, s2):
    if len(s1) != len(s2):
        return False

    s1_sorted = sorted(s1)
    s2_sorted = sorted(s2)

    for i in range(len(s1)):
        if s1_sorted[i] != s2_sorted[i]:
            return False

    return True

# chat GPT

def is_permutation(s1, s2):
    
    if len(s1) != len(s2):
        return False

    s1_sorted = sorted(s1)
    s2_sorted = sorted(s2)

    return s1_sorted == s2_sorted   # ���ĵ� ���ڿ��� ���� 



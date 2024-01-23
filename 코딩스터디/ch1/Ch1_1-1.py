# 문자열이 주어졌을 때 같은 문자가 중복되어 등장하는지 확인하는 알고리즘 

# 해시테이블
def duplicate_char (str):

    if len(str) > 128 :
        return False
    
    char_set = set()            # 집합으로 선언
    for char in str:            # 현재 문자가 char_set에 존재하는지 확인 
        if char in char_set:
            return False
        char_set.add(char)      # 현재 문자가 char_set에 없다면 char_set에 추가
    return True

# 자료구조 없이
def duplicate_char (str) :

    if len(str) > 128 :
        return False
    
    for i in range(len(str)) :
        for j in range(i+1,len(str)):
            if str[i] == str[j]:
                return False
    return True 



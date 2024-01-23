def is_one_edit_away(str1, str2):
    def replace(s1, s2):   
        diff = False    #변수 diff를 False로 초기화
        for c1, c2 in zip(s1, s2):  #각 문자를 순서대로 비교하는 for 루프
            if c1 != c2:    #현재 비교중인 두 문자가 다르면
                if diff:  #변수 diff가 True면 이미 차이가 발견됐다는거니까
                    return False    #False
                diff = True   #차이가 처음 발견됨 = diff에 True 저장 
        return True     

    def insert(s1, s2):
        index1 = index2 = 0     #index1,2 변수를 0으로 초기화
        while index1 < len(s1) and index2 < len(s2):    #s1과 s2의 인덱스가 각 문자열 길이를 넘지 않는동안 반복 
            if s1[index1] != s2[index2]:    #현재 인덱스에서 s1,s2다르면
                if index1 != index2:    #두 인덱스가 다르면(이미삽입,삭제 이뤄진거면)
                    return False    #False 반환
                index2 += 1     #s2의 인덱스만 증가시켜 다음문자로
            else:   #현재 인덱스에서 s1,s2같으면
                index1 += 1     #s1의 인덱스 증가
                index2 += 1     #s2의 인덱스 증가
        return True     

    if len(str1) == len(str2):      #두 문자열의 길이가 같으면
        return replace(str1, str2)     #교체 함수에 넣어
    elif len(str1) + 1 == len(str2):    #str1이 str2보다 1 크면 (삭제)
        return insert(str1, str2)  #str1이 s1으로
    elif len(str1) - 1 == len(str2):    #str1이 str2보다 1작으면 (삽입)
        return insert(str2, str1)  #str1이 s2로
    return False






#one_edit_replace 함수는 두 문자열 s1과 s2가 동일한 길이일 때, 
#한 번의 교체로 서로 같아질 수 있는지 확인합니다. 
#diff 변수는 이전에 문자가 달랐던 경우를 표시합니다. 
#문자가 처음으로 다른 경우에는 diff를 True로 설정하고, 두 번째로 다른 문자가 나타나면 False를 반환합니다.

#one_edit_insert 함수는 한 문자열이 다른 문자열보다 
#정확히 한 글자 더 길 때 사용됩니다. 
#이 함수는 두 문자열 사이에 정확히 한 글자의 차이만 있는지 확인합니다. 
#인덱스가 서로 다를 때 False를 반환하면, 이는 이미 한 번의 삽입 또는 삭제가 이루어졌음을 의미합니다.
def is_palindrome_permutation(s):
    # 문자열 전처리
    s = ''.join(filter(str.isalnum, s.lower()))
    
    # 각 문자의 등장 횟수를 세기
    char_count = {}
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # 홀수 번 나타나는 문자의 개수 확인
    odd_count = 0
    for count in char_count.values():
        if count % 2 != 0:
            odd_count += 1
            if odd_count > 1:
                return False

    return True



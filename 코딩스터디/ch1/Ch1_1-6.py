def compress_string(s):
    if not s:   #문자열이 비어있을때 처리하는거
        return ""

    compressed = []
    count = 1   #현재 문자의 반복횟수
    for i in range(1, len(s)):  #문자열의 2번째부터 마지막 문자 이전까지 순회
        if s[i] == s[i - 1]:
            count += 1  #동일한 문자 반복하면 count에 1증가시켜
        else:
            compressed.append(s[i - 1] + str(count))    
            count = 1
    compressed.append(s[-1] + str(count))

    compressed_str = ''.join(compressed)    #리스트의 모든 요소를 하나의 문자열로 결합
    return compressed_str

def compress_good(comp_str, orig_str):
    if len(comp_str) > len(orig_str):
        return orig_str
    else:
        return comp_str
    
# 사용 예시
original_string = "aabccccaaa"
compressed_string = compress_string(original_string)
result = compress_good(compressed_string, original_string)
print(result)
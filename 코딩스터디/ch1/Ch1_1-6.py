def compress_string(s):
    if not s:   #���ڿ��� ��������� ó���ϴ°�
        return ""

    compressed = []
    count = 1   #���� ������ �ݺ�Ƚ��
    for i in range(1, len(s)):  #���ڿ��� 2��°���� ������ ���� �������� ��ȸ
        if s[i] == s[i - 1]:
            count += 1  #������ ���� �ݺ��ϸ� count�� 1��������
        else:
            compressed.append(s[i - 1] + str(count))    
            count = 1
    compressed.append(s[-1] + str(count))

    compressed_str = ''.join(compressed)    #����Ʈ�� ��� ��Ҹ� �ϳ��� ���ڿ��� ����
    return compressed_str

def compress_good(comp_str, orig_str):
    if len(comp_str) > len(orig_str):
        return orig_str
    else:
        return comp_str
    
# ��� ����
original_string = "aabccccaaa"
compressed_string = compress_string(original_string)
result = compress_good(compressed_string, original_string)
print(result)
# ���ڿ��� �־����� �� ���� ���ڰ� �ߺ��Ǿ� �����ϴ��� Ȯ���ϴ� �˰��� 

# �ؽ����̺�
def duplicate_char (str):

    if len(str) > 128 :
        return False
    
    char_set = set()            # �������� ����
    for char in str:            # ���� ���ڰ� char_set�� �����ϴ��� Ȯ�� 
        if char in char_set:
            return False
        char_set.add(char)      # ���� ���ڰ� char_set�� ���ٸ� char_set�� �߰�
    return True

# �ڷᱸ�� ����
def duplicate_char (str) :

    if len(str) > 128 :
        return False
    
    for i in range(len(str)) :
        for j in range(i+1,len(str)):
            if str[i] == str[j]:
                return False
    return True 



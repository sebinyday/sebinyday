# ���ڿ��� ����ִ� ��� ������ ��%20�� ���� �ٲ��ִ� �޼���
# ���ڿ��� ���� ���̰� �Բ� �־����ٰ� �����ص� ��  

# �� Ǯ�� (Ʋ��)

def URL_conversion(str):
    for i in range (len(str)):
        if str[i] == " ":    
            str[i] = "%20"  # ���̽��� ���ڿ��� immutable(�Һ�)�ϱ� ������ 
                            # ���� ���ڸ� �ε����� ���� ���� �����ϴ� ���� �Ұ���

    return str 

# chat GPT

def URL_conversion(string):
    return string.replace(" ", "%20")       #replace() �޼���� ���� ���ڿ��� ��� �ν��Ͻ��� ��ü



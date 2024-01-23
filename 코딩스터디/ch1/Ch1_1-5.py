def is_one_edit_away(str1, str2):
    def replace(s1, s2):   
        diff = False    #���� diff�� False�� �ʱ�ȭ
        for c1, c2 in zip(s1, s2):  #�� ���ڸ� ������� ���ϴ� for ����
            if c1 != c2:    #���� ������ �� ���ڰ� �ٸ���
                if diff:  #���� diff�� True�� �̹� ���̰� �߰ߵƴٴ°Ŵϱ�
                    return False    #False
                diff = True   #���̰� ó�� �߰ߵ� = diff�� True ���� 
        return True     

    def insert(s1, s2):
        index1 = index2 = 0     #index1,2 ������ 0���� �ʱ�ȭ
        while index1 < len(s1) and index2 < len(s2):    #s1�� s2�� �ε����� �� ���ڿ� ���̸� ���� �ʴµ��� �ݺ� 
            if s1[index1] != s2[index2]:    #���� �ε������� s1,s2�ٸ���
                if index1 != index2:    #�� �ε����� �ٸ���(�̹̻���,���� �̷����Ÿ�)
                    return False    #False ��ȯ
                index2 += 1     #s2�� �ε����� �������� �������ڷ�
            else:   #���� �ε������� s1,s2������
                index1 += 1     #s1�� �ε��� ����
                index2 += 1     #s2�� �ε��� ����
        return True     

    if len(str1) == len(str2):      #�� ���ڿ��� ���̰� ������
        return replace(str1, str2)     #��ü �Լ��� �־�
    elif len(str1) + 1 == len(str2):    #str1�� str2���� 1 ũ�� (����)
        return insert(str1, str2)  #str1�� s1����
    elif len(str1) - 1 == len(str2):    #str1�� str2���� 1������ (����)
        return insert(str2, str1)  #str1�� s2��
    return False






#one_edit_replace �Լ��� �� ���ڿ� s1�� s2�� ������ ������ ��, 
#�� ���� ��ü�� ���� ������ �� �ִ��� Ȯ���մϴ�. 
#diff ������ ������ ���ڰ� �޶��� ��츦 ǥ���մϴ�. 
#���ڰ� ó������ �ٸ� ��쿡�� diff�� True�� �����ϰ�, �� ��°�� �ٸ� ���ڰ� ��Ÿ���� False�� ��ȯ�մϴ�.

#one_edit_insert �Լ��� �� ���ڿ��� �ٸ� ���ڿ����� 
#��Ȯ�� �� ���� �� �� �� ���˴ϴ�. 
#�� �Լ��� �� ���ڿ� ���̿� ��Ȯ�� �� ������ ���̸� �ִ��� Ȯ���մϴ�. 
#�ε����� ���� �ٸ� �� False�� ��ȯ�ϸ�, �̴� �̹� �� ���� ���� �Ǵ� ������ �̷�������� �ǹ��մϴ�.
# 문자열에 들어있는 모든 공백을 ‘%20’ 으로 바꿔주는 메서드
# 문자열의 최종 길이가 함께 주어진다고 가정해도 됨  

# 내 풀이 (틀림)

def URL_conversion(str):
    for i in range (len(str)):
        if str[i] == " ":    
            str[i] = "%20"  # 파이썬의 문자열은 immutable(불변)하기 때문에 
                            # 개별 문자를 인덱스를 통해 직접 변경하는 것은 불가능

    return str 

# chat GPT

def URL_conversion(string):
    return string.replace(" ", "%20")       #replace() 메서드는 원본 문자열의 모든 인스턴스를 대체



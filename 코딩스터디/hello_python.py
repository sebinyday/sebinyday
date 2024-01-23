print(1 - 2)  # -1 
print(5 * 6)  # 30
print(7 / 5)  # 1.4 
print(5 ** 2) # 25

print( type(15) )       # <class 'int'>
print( type(2.65) )     # <class 'float'>
print( type("good") )   # <class 'str'>

x = 10
print( x )      # 10

y = 0.314
print( x * y )  # 3.14

a = [2,4,6,8,10]
print( a )      # [2,4,6,8,10]
print( a[0] )   # 2
print( len(a) ) # 5

a[3] = 7
print( a )      # [2,4,6,7,10]

a = [2,4,6,8,10]
print( a[0:3] ) # [2, 4, 6]
print( a[:3] )  # [2, 4, 6]
print( a[2:] )  # [6, 8, 10]
print( a[:-3] ) # [2, 4]

me = {'height':160} 
me['height']        #160 원소에 접근하여 값 출력
me['weight'] = 50   # 새 원소 추가 
print(me)           #{'height': 160, 'weight': 50}


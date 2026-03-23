
import random
def randomText(textArr,m,n):
    length = len(textArr)
    if length < 1:
        return ''
    if length == 1:
        return textArr[0]

    randomNumber= random.randint(0, length - 1)
    print(randomNumber)
    if randomNumber!=m and randomNumber!=n:
       print('randomNumber',randomNumber)

       return randomNumber

    return randomText(textArr,m,n)
randStr = [0,1,2,3,4]

randN=randomText(randStr,0,1)
print(randN)
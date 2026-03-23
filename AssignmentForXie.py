def caluateRM(w1, w2, x1, x2, l):
    r1 = (w1 * (l - x1) + w2 * (l - x2)) / l
    r2 = (w1 * x1 + w2 * x2) / l
    m1 = -((w1 * x1 * (l - x1) * (l - x1)) / (l * l) + (w2 * x2 * (l - x2) * (l - x2)) / (l * l))
    m2 = (w1 * x1 * x1 * (l - x1)) / (l * l) + (w2 * x2 * x2 * (l - x2)) / (l * l)
    return r1, r2, m1, m2


if __name__ == '__main__':
    w1=1
    w2=2
    for i in range(10):
        r1, r2, m1, m2 = caluateRM(w1, w2, 3, 4, 2)
        w1+=1
        w2+=1
        print(r1, r2, m1, m2)


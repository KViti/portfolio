def Win(k, n):
    win=0
    i=1
    while k>(n+1):
        m=k%n
        print('m:', m)
        k=k//n
        print('k:', k)
        win += m*n**i
        print('win:', win)
        i+=1
    print('win:', win)
    k = k // n
    win=win+k
    return (win)


k, n = int(input()), int(input())
print(Win(k, n))

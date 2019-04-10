def map_feature(x1, x2, num):
    f = open('ex2data3.txt','w')
    A_list = [x1,x2]
    for i in range(num):
        for j in range(num):
            if i+j<=6:
                A_list.append((x1**i)*(x2**j))

    print(A_list)
    print(len(A_list))
    f.write(str(A_list))
    f.close()



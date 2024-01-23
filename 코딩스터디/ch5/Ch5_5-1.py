def insertMintoN(N, M, i, j):

    allOnes = ~0  
    left = allOnes << (j + 1)  
    right = ((1 << i) - 1)  
    mask = left | right  

    N &= mask

    M <<= i

    return N | M

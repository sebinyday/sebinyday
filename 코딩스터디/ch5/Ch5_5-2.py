def float_to_binary(num):
    if num >= 1 or num <= 0:
        return "ERROR"

    binary = "."
    while num > 0:
        
        if len(binary) > 32:
            return "ERROR"

        r = num * 2
        if r >= 1:
            binary += "1"
            num = r - 1
        else:
            binary += "0"
            num = r

    return binary


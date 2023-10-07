def print_pattern(string):
    string += string
    loops = (len(string)-2)/2
    count = 0
    while loops >= 0:
        tmp = "_" * (count) + string[count:len(string)-count]
        print(tmp)
        loops -= 1
        count += 1

print_pattern(input())
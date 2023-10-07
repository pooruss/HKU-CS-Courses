count = 0
total = 0

while True:
    num = int(input(""))
    if num == 0:
        break
    count += 1
    total += num

average = total // count

print(average)
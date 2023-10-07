n = int(input(""))

fibonacci_numbers = [0, 1]

for i in range(2, n):
    fibonacci_numbers.append(fibonacci_numbers[i-1] + fibonacci_numbers[i-2])

for i in range(len(fibonacci_numbers)):
    if i == n:
        break
    if i == 0:
        print(fibonacci_numbers[i], end='')
    else:
        print(" " + str(fibonacci_numbers[i]), end='')
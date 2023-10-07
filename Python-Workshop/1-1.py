x = int(input())
if x % 4 != 0 or x > 0:
  print('apple', end='')
  if x < 0:
    print('banana', end='')
elif x % 4 == 0:
  print('cherry', end='')
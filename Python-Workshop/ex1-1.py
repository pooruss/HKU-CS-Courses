import numpy as np

def generate_sequence(a, b):
    a = float(a)
    b = float(b)
    sequence = np.linspace(a, b, 9)
    sequence_array = sequence.reshape(3, 3).astype(int)
    return sequence_array

# Example usage:
a = 0
b = 0
for i in range(2):
    if i == 0:
        a = input()
    else:
        b = input()
sequence = generate_sequence(a, b)
print(sequence)
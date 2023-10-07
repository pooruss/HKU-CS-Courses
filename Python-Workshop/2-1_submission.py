def read_integers():
    integers = []
    while True:
        num = int(input(""))
        if num == 0:
            break
        integers.append(num)
    return tuple(integers)

# Function to repeatedly remove the first and last item from the tuple and print it
def remove_print_tuple(t):
    while len(t) > 0:
        first = t[0]
        last = t[-1]
        t = t[1:-1]
        a_tuple = ()
        for i in range(first, last+1):
            a_tuple = a_tuple + (i,)
        print(a_tuple)

# Main program
if __name__ == "__main__":
    # Read integers
    my_tuple = read_integers()

    # Remove the first and last item from the tuple and print
    remove_print_tuple(my_tuple)
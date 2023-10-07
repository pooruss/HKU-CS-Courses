def triangle_type(a, b, c):
    if a + b <= c or a + c <= b or b + c <= a:
        return "N"  # Not a triangle
    
    if a == b == c:
        return "E"  # Equilateral triangle
    
    if a == b or a == c or b == c:
        return "I"  # Isosceles triangle
    
    if a**2 + b**2 == c**2 or a**2 + c**2 == b**2 or b**2 + c**2 == a**2:
        return "R"  # Right triangle
    
    return "S"  # Scalene triangle

# Read input from user
items = input()
a, b, c = items.split(" ")
a = int(a)
b = int(b)
c = int(c)
# Determine triangle type and print the result
triangle_type_output = triangle_type(a, b, c)
print(triangle_type_output)
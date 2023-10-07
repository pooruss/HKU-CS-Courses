def seconds_to_long_format(duration):
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    result = f"{hours}h {minutes}m {seconds}s"
    return result

# Example usage
inputs = float(input())
print(seconds_to_long_format(inputs))
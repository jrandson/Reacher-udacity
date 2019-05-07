import time
print("Hello")

for i in range(10):
    print("\r{}/10".format(i), end="", flush=True)
    time.sleep(1)

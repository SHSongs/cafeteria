# import json

# # cnt = b'{"cnt_in": 3, "entrance": "in"}'
# # my_json = json.loads(cnt)
# # print(my_json['entrance'])

# current = 0

# def current_up():
#     global current
#     return current

# current_up()

# print(current)

import time

for i in range(10):
    file = open("새.txt", 'a')
    file.write("a")
    file.close()
    time.sleep(1)
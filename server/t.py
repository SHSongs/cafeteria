import json

# cnt = b'{"cnt_in": 3, "entrance": "in"}'
# my_json = json.loads(cnt)
# print(my_json['entrance'])

current = 0

def current_up():
    global current
    return current

current_up()

print(current)
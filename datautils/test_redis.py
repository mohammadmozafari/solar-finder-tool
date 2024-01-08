import redis

r = redis.Redis(host='127.0.0.1', port=6379)
print(r.ping())

print(r.hset('job:1', 'name', 'mnch'))

print(list(r.scan_iter('*')))
print(list(r.scan_iter('job:*')))

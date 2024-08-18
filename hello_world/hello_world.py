import cohere
import json

with open('../configs.json') as jf:
	configs = json.load(jf)
	jf.close()
api_key = configs.get('api-keys', dict()).get('trial')
co = cohere.Client(api_key)

response = co.chat(
	message="hello world!"
)

print(type(response))

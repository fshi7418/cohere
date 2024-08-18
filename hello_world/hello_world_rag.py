import cohere
import json

# basic setup
with open('../configs.json') as jf:
	configs = json.load(jf)
	jf.close()
api_key = configs.get('api-keys', dict()).get('trial')
co = cohere.Client(api_key)

# setup
# recommended length of each document is 300 words or fewer
documents = [
	{
		'title': 'Fact 1',
		'text': 'I am currently reading Infinite Jest.'
	},
	{
		'title': 'Fact 2',
		'text': 'I recently finished reading Two Years Before the Mast.'
	},
]
message = 'What books have I recently read?'

# generate response
response = co.chat_stream(
	message=message,
	documents=documents
)

# display response
citations = []
cited_documents = []

for event in response:
	if event.event_type == "text-generation":
		print(event.text, end="")
	elif event.event_type == "citation-generation":
		citations.extend(event.citations)
	elif event.event_type == "stream-end":
		cited_documents = event.response.documents

# Display the citations and source documents
if citations:
	print("\n\nCITATIONS:")
	for citation in citations:
		print(citation)

	print("\nDOCUMENTS:")
	for document in cited_documents:
		print(document)

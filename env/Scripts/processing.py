from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


# Create a pipeline for sentiment analysis
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define your input text
text = "I love using Hugging Face models! They are so easy to use."

# Get the sentiment
result = nlp(text)

# Output the result
print(result)

# Now you can use 'model' and 'tokenizer' in your code

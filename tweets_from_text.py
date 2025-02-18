import re
from mistralai.client import MistralClient
from typing import Generator

def chunks_from_text(text: str, chunk_size: int=1024, overlap_size: int=256):
    def split_into_paragraphs(text):
        return re.split(r'\n{2,}', text.strip())

    def split_into_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def estimate_token_count(text):
        return len(text.split())

    segments = []
    for paragraph in split_into_paragraphs(text):
        sentence_segments = split_into_sentences(paragraph)
        if sentence_segments:
            segments.extend(sentence_segments)
            segments.append("\n\n")

    chunks = []
    i = 0
    while i < len(segments):
        current_chunk_segments = []
        current_token_count = 0
        j = i

        while j < len(segments) and current_token_count < chunk_size:
            current_chunk_segments.append(segments[j])
            current_token_count += estimate_token_count(segments[j])
            j += 1

        chunk_text = " ".join(current_chunk_segments).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if j >= len(segments):
            break

        overlap_tokens = 0
        new_start = j
        for k in range(j - 1, i - 1, -1):
            overlap_tokens += estimate_token_count(segments[k])
            if overlap_tokens >= overlap_size:
                new_start = k
                break

        if new_start == i:
            new_start = min(i + 1, len(segments) - 1)

        i = new_start

    return chunks

def tweets_from_chunks(chunks: list[str], client: MistralClient, model: str) -> Generator[str, None, None]:
    def extract_tweets(text):
        tweets = re.findall(r'<tweet>(.*?)</tweet>', text, flags=re.DOTALL)
        return [tweet.strip() for tweet in tweets if tweet.strip()]
    
    seen_tweets = set()
    all_tweets = []
    
    for chunk in chunks:
        user_message = {
            "role": "user",
            "content": (
                "# Role:\n"
                "You're a literary social media strategist creating viral Twitter threads from classic poetry.\n\n"

                "# Instructions:\n"
                "1. ANALYZE the text for key narrative elements and emotional arcs\n"
                "2. CREATE engaging tweets\n"
                "3. FORMAT each tweet as: <tweet>[content]</tweet>\n\n"

                "# Rules:\n"
                "- Maintain chronological story flow between tweets\n"
                "- Use simple, impactful language\n"
                "- Use emojis sparingly\n"
                "- Employ comedy when appropriate\n"
                "- Place hashtags at end, mix popular and niche tags\n"
                "- Vary tweet structures (questions, statements, dialog)\n"
                "- Avoid repetition unless it is critical to the original text\n"
                "- Keep in mind that the text may overlap with previous output if present\n\n"

                "# Critical Reminders:\n"
                "- STRICTLY follow the <tweet></tweet> format\n"
                "- PENALTY for markdown/formatting errors\n"
                "- AVOID repeating similar phrases or concepts\n"
                "- Each tweet must advance the narrative\n\n"

                f"{'# Previous Output:\n' + '\n'.join([f'<tweet>{t}</tweet>' for t in all_tweets[-10:]]) + '\n\n' if all_tweets else ''}"

                "# Text to Convert:\n"
                f"{chunk}"
            )
        }
        
        response = client.chat.complete(
            model=model,
            messages=[user_message],
            temperature=1.0,
            top_p=0.5,
            presence_penalty=1.0,
            frequency_penalty=0.25,
        )

        generated_text = response.choices[0].message.content
        extracted_tweets = extract_tweets(generated_text)
        
        new_tweets = 0
        for tweet in extracted_tweets:
            if tweet not in seen_tweets:
                seen_tweets.add(tweet)
                all_tweets.append(tweet)
                new_tweets += 1
                yield tweet

def tweets_from_text(text: str, chunk_size: int, overlap_size: int, client: MistralClient, model: str):
    chunks = chunks_from_text(text, chunk_size, overlap_size)
    tweet_generator = tweets_from_chunks(chunks, client, model)
    tweets = []
    for tweet in tweet_generator:
        tweets.append(tweet)
    return tweets
import logging
from typing import List

from config import GPT_MODEL, openai_client

logger = logging.getLogger(__name__)

def ask_gpt(question: str, relevant_texts: List[str]) -> str:
    context = "\n\n".join([f"Document {idx + 1}: {text}" for idx, text in enumerate(relevant_texts[:7])])
    prompt = f"Relevant internal documentation:\n\n{context}\n\nBased on the internal documentation, answer the question: {question}"

    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are Senior member of the Coupa Support team, dedicated to providing comprehensive and accurate assistance to new-hires. You provide answers with intent to instruct and prevent the same question from being asked again by the same person. Primarily base your answers on the information provided, using external sources to fill in the gaps."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Failed to execute OpenAI API call: {str(e)}")
        return "Error in processing the request. Please check the logs."

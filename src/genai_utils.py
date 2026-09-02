"""GenAI Sentiment Analysis Utility for Yelp Reviews.

Includes automatic compression support, checkpointing, rate-limit recovery,
capped backoff delays, item-level fallback, and Groq JSON mode.
"""

import json        #for working with JSON data
import logging     #to display messages about what the program is doing
import os          #to interact with the operating system
import re          #regular expressions: to split a review into sentences
import time
from typing import Any, Dict, List
import pandas as pd
from pydantic import BaseModel, Field    #Pydantic allows us to define what the output is supposed to look like.

# Setup logger
# A logger is basically a system for recording messages about what your program is doing.
logger = logging.getLogger(__name__)    #__name__ identifies the current Python module.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# IF a Yelp review is very long, shorten it before sending it to the GenAI model, so we use fewer tokens while 
# retaining information from the beginning and end of the review.

# --- Text Compression Helper ---
def compress_review(text: str, max_words: int = 120) -> str:     
  """Extracts first 2 and last 2 sentences for long reviews to reduce tokens."""
  if not text or not isinstance(text, str):         #isinstance(): Checks is this object of a particular type?
    return ""

  words = text.split()
  if len(words) <= max_words:
    return text
                              #r"(?<=[.!?]) +" : It is a regular expression (regex)
                              #?<= : Positive lookbehind: Check what comes immediately before this position, but don't include that thing in the match."
                              #(?<=[.!?]) : "Look behind me and make sure there's a ., !, or ?."
                              #+ : Look for one or more spaces.
  sentences = [
      s.strip() for s in re.split(r"(?<=[.!?]) +", text) if s.strip()
  ]
  if len(sentences) <= 3:
    return " ".join(words[:max_words]) + "..."

  return " ".join(sentences[:2]) + " ... " + " ".join(sentences[-2:])


# --- Data Output Schema ---
# How do we make sure the model gives us the information in the format we expect?

class SingleReviewSentiment(BaseModel):
  review_id: str = Field(description="Original Yelp review ID preserved exactly.")
  sentiment_label: str = Field(
      description="Must be: positive, neutral, or negative."
  )
  sentiment_score: float = Field(
      description="Number from -1.0 (very negative) to 1.0 (very positive)."
  )
  sentiment_reason: str = Field(description="Short explanation of the rating.")


#An API key is essentially a credential that allows your program to access an API.
# --- Client Initialization ---
def get_groq_client():
  api_key = os.getenv("GROQ_API_KEY")
  if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not found!")

  try:                      #try: Python's error-handling mechanism.
    from groq import Groq

    return Groq(api_key=api_key)
  except ImportError:
    raise ImportError("The 'groq' package is not installed.")


# --- Prompt Builder ---
def build_prompt(reviews: List[dict]) -> str:
  reviews_json = json.dumps(reviews, indent=2)
  expected_count = len(reviews)

  return f"""You are an expert customer sentiment analyst evaluating merchant reviews.

Analyze each of the following Yelp reviews independently and extract structured sentiment data.

### Input Reviews ({expected_count} items):
{reviews_json}

### Instructions & Rules:
1. Preserve the exact `review_id` string for every review.
2. Return EXACTLY ONE result entry for every input review provided ({expected_count} total). Do not skip or omit any review.
3. `sentiment_label`: Classify strictly as one of ["positive", "neutral", "negative"].
4. `sentiment_score`: Continuous numerical score bounded between -1.0 (extremely negative) and +1.0 (extremely positive).
5. `sentiment_reason`: Provide a concise explanation (maximum 15 words) for the sentiment assignment.
6. Output MUST strictly be valid raw JSON with a top-level "results" array.

JSON Format:
{{
  "results": [
    {{
      "review_id": "string",
      "sentiment_label": "positive|neutral|negative",
      "sentiment_score": 0.0,
      "sentiment_reason": "string"
    }}
  ]
}}
"""


# --- Batching Utility ---
def batch_reviews(
    reviews: List[dict], batch_size: int = 5
) -> List[List[dict]]:
  return [
      reviews[i : i + batch_size] for i in range(0, len(reviews), batch_size)
  ]  


# --- Validation Helper ---
def validate_sentiment_result(
    raw_results: List[dict], expected_ids: List[str]
) -> List[dict]:
  validated_output = []
  returned_ids = set()
  valid_labels = {"positive", "neutral", "negative"}

  for item in raw_results:
    parsed = SingleReviewSentiment(**item)

    if parsed.sentiment_label not in valid_labels:
      raise ValueError(
          f"Invalid label '{parsed.sentiment_label}' for review_id"
          f" '{parsed.review_id}'"
      )

    if not (-1.0 <= parsed.sentiment_score <= 1.0):
      raise ValueError(
          f"Score {parsed.sentiment_score} outside [-1.0, 1.0] bound for"
          f" review_id '{parsed.review_id}'"
      )

    returned_ids.add(parsed.review_id)
    validated_output.append(parsed.model_dump())

  missing_ids = set(expected_ids) - returned_ids
  if missing_ids:
    raise ValueError(
        f"Model omitted response for {len(missing_ids)} review_id(s):"
        f" {missing_ids}"
    )

  return validated_output


# --- Single Batch Execution with Robust Retry & Item-Level Fallback ---
# it deals execution and with what happens when the GenAI API fails or gives a problematic response.
# Try to process the five reviews together. If the request fails, retry it. If the batch still can't be processed, 
# fall back to processing the reviews one at a time so that one problematic batch doesn't destroy the entire analysis."

def analyze_review_batch(
    client,
    batch: List[dict],
    model: str = "openai/gpt-oss-20b",  
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0
) -> List[dict]:
  expected_ids = [r["review_id"] for r in batch]
  prompt = build_prompt(batch)

  for attempt in range(1, max_retries + 1):
    try:
      completion = client.chat.completions.create(
          model=model,
          messages=[{"role": "user", "content": prompt}],
          temperature=0.0,      #I used a temperature of 0 to make the sentiment classification as deterministic and consistent as possible."
          max_tokens=2048,
          response_format={"type": "json_object"},
      )

      raw_content = completion.choices[0].message.content.strip()

      if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
          raw_content = raw_content[4:].strip()
      if raw_content.endswith("```"):
        raw_content = raw_content[:-3].strip()

      parsed_json = json.loads(raw_content)
      results_list = parsed_json.get(
          "results", parsed_json if isinstance(parsed_json, list) else []
      )

      return validate_sentiment_result(results_list, expected_ids)

    except Exception as e:
      logger.warning(
          f"Attempt {attempt}/{max_retries} failed for batch (size"
          f" {len(batch)}): {e}"
      )
      if attempt == max_retries:
        # Fallback: process problematic items individually 1-by-1 if batch processing repeatedly fails
        if len(batch) > 1:
          logger.info("Splitting problematic batch into individual items...")
          single_results = []
          for single_item in batch:
            try:
              res = analyze_review_batch(
                  client, [single_item], model=model, max_retries=2
              )
              single_results.extend(res)
            except Exception:
              logger.error(
                  "Fallback triggered for unparseable review:"
                  f" {single_item['review_id']}"
              )
              single_results.append({
                  "review_id": single_item["review_id"],
                  "sentiment_label": "neutral",
                  "sentiment_score": 0.0,
                  "sentiment_reason": "Fallback assigned due to parsing error.",
              })
          return single_results
        raise e

      # Capped backoff delay so wait times never explode
      sleep_time = min(backoff_factor**attempt, max_delay)
      time.sleep(sleep_time)


# --- Resumable Main Public Interface ---
def analyze_reviews(
    df_reviews: pd.DataFrame,
    text_col: str = "compressed_text",
    batch_size: int = 5,
    delay_between_batches: float = 2,
    checkpoint_file: str = "../../data/raw/yelp/yelp_review_sentiment_checkpoint.csv",
    model: str = "openai/gpt-oss-20b"   
) -> pd.DataFrame:

  df_work = df_reviews.copy()

  # Automatically generate compressed text column if omitted
  if text_col not in df_work.columns:
    if "text" in df_work.columns:
      logger.info(f"Column '{text_col}' not found. Auto-compressing 'text'...")
      df_work[text_col] = df_work["text"].apply(
          lambda x: compress_review(str(x), max_words=120)
      )
    else:
      raise KeyError(
          f"Input DataFrame missing required column '{text_col}' or 'text'."
      )

  # Check checkpoint
  completed_ids = set()
  completed_results = []

  if os.path.exists(checkpoint_file):
    df_checkpoint = pd.read_csv(checkpoint_file)
    completed_ids = set(df_checkpoint["review_id"])
    completed_results = df_checkpoint.to_dict(orient="records")
    logger.info(
        "Found existing checkpoint file! Resuming with"
        f" {len(completed_ids)} reviews already completed."
    )

  df_remaining = df_work[~df_work["review_id"].isin(completed_ids)].copy()

  if len(df_remaining) == 0:
    logger.info(
        "All reviews have already been processed in the checkpoint file!"
    )
    return pd.DataFrame(completed_results)

  client = get_groq_client()

  records = [
      {"review_id": row["review_id"], "text": row[text_col]}
      for _, row in df_remaining.iterrows()
  ]

  batches = batch_reviews(records, batch_size=batch_size)
  total_batches = len(batches)

  logger.info(
      f"Processing remaining {len(records)} reviews in {total_batches}"
      " batches..."
  )

  for i, batch in enumerate(batches, start=1):
    batch_res = analyze_review_batch(client, batch, model=model)
    completed_results.extend(batch_res)

    df_temp = pd.DataFrame(completed_results)
    df_temp.to_csv(checkpoint_file, index=False)

    if i % 10 == 0 or i == total_batches:
      logger.info(
          f"Batch {i}/{total_batches} complete"
          f" ({len(completed_results)}/{len(df_work)} total saved)."
      )

    if delay_between_batches > 0 and i < total_batches:
      time.sleep(delay_between_batches)

  return pd.DataFrame(completed_results)